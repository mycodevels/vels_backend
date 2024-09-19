from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import firebase_admin
from firebase_admin import credentials, firestore
import os
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

import pdfplumber
import re
import nltk
from werkzeug.utils import secure_filename
app = FastAPI()

# Firebase initialization
cred = credentials.Certificate("vels-d0547-firebase-adminsdk-6kni4-2d374ae1f1.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
USERS_COLLECTION = "users"
VOTERS_COLLECTION ="Voters"
# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload folder exists
os.makedirs('uploads', exist_ok=True)

class SigninData(BaseModel):
    email: str
    password: str

class SignupData(BaseModel):
    username: str
    email: str
    password: str
    selectedMode: str
    contact: str

@app.post('/signin')
def signin(data: SigninData):
    # Extract data from the request
    email = data.email
    password = data.password
    
    if not email or not password:
        return JSONResponse(content={"error": "Missing data"}, status_code=400)
    
    # Fetch user from Firestore
    users_ref = db.collection(USERS_COLLECTION)  # Assuming collection name is 'users'
    query = users_ref.where(u'email', u'==', email).limit(1).get()

    if not query:
        return JSONResponse(content={"error": "User not found"}, status_code=404)
    
    user_doc = query[0].to_dict()

    # Check if the provided password matches the stored password
    if user_doc.get('password') == password:
        return JSONResponse(content={"message": "Sign in successful", "user": user_doc}, status_code=200)
    else:
        return JSONResponse(content={"error": "Invalid credentials"}, status_code=401)

@app.post('/signup')
def signup(data: SignupData):
    # Extract data from the request
    username = data.username
    email = data.email
    password = data.password
    selectedMode = data.selectedMode
    contact = data.contact
    
    if not username or not email or not password:
        return JSONResponse(content={"error": "Missing data"}, status_code=400)
    
    users_ref = db.collection(USERS_COLLECTION)  
    new_user_ref = users_ref.document()  
    new_user_ref.set({
        'username': username,
        'email': email,
        'password': password,
        'contact': contact,
        'selectedMode': selectedMode
    })
    
    # Return success message
    return JSONResponse(content={"message": "Signup successful"}, status_code=201)


@app.get('/users')
def get_all_users():
    try:
        users_ref = db.collection(USERS_COLLECTION)  # Replace with your actual collection name
        docs = users_ref.stream()

        users = []
        for doc in docs:
            user_data = doc.to_dict()
            user_data['id'] = doc.id  # Add document ID to the data
            users.append(user_data)

        return JSONResponse(content=users, status_code=200)
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Failed to retrieve users"}, status_code=500)


class AddressUpdateRequest(BaseModel):
    addressline2: str

@app.put('/users/voters/{precinct_id}/{document_id}')
def update_voter_address(precinct_id: str, document_id: str, data: AddressUpdateRequest):
    try:
        addressline2 = data.addressline2
        
        if not addressline2:
            return JSONResponse(content={"error": "addressline2 is required"}, status_code=400)

        # Firestore document path
        doc_ref = db.collection(VOTERS_COLLECTION).document(precinct_id).collection('voters').document(document_id)

        # Update the document by adding the new field 'addressline2'
        doc_ref.update({
            'addressline2': addressline2
        })

        return JSONResponse(content={"message": "Address updated successfully"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get('/users/{doc_id}/voters')
def get_voters_by_documentid(doc_id: str):
    try:
        # Reference to the specific user document
        user_ref = db.collection(VOTERS_COLLECTION).document(doc_id)  
        
        # Fetch the user document
        user_doc = user_ref.get()

        if user_doc.exists:
            # Reference to the voters sub-collection
            voters_ref = user_ref.collection('voters')
            
            # Fetch all voter documents
            docs = voters_ref.stream()

            voters = []
            for doc in docs:
                voter_data = doc.to_dict()
                voter_data['id'] = doc.id  # Add document ID to the voter data
                
                # Optional voter fields, defaulting to a message if not found
                voter_info = {
                    "id": voter_data.get("id", "ID not found"),
                    "fullName": voter_data.get("Full Name", "Full Name not found"),
                    "voterNo": voter_data.get("Voter No", "Voter No not found"),
                    "address": voter_data.get("Address", "Address not found"),
                    "barangay": voter_data.get("Barangay", "Barangay not found"),
                    "city": voter_data.get("City", "City not found"),
                    "province": voter_data.get("Province", "Province not found"),
                    "addressline2": voter_data.get("addressline2", "")
                }

                voters.append(voter_info)

            return JSONResponse(content=voters, status_code=200)
        else:
            return JSONResponse(content={"error": "User not found"}, status_code=404)
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Failed to retrieve voters"}, status_code=500)



def extract_voter_information_from_pdf(pdf_path):
    # Initialize variables
    precinct_voters = {}
    results = {}

    # Regular expressions to extract precinct, city, province, and barangay
    prec_regex = re.compile(r'Prec\s*:\s*(\d+\w?)')
    city_regex = re.compile(r'CITY / MUNICIPALITY : (.+)')
    province_regex = re.compile(r'PROVINCE : (.+)')
    barangay_line_regex = re.compile(r'BARANGAY : (.+)')
    
    # Regex to extract names with an asterisk followed by a space
    asterisk_name_regex = re.compile(r'\*\s+([A-Z]+(?:\s[A-Z]+)*)\s+([A-Z]+(?:\s[A-Z]+)*)\s+([A-Z]+(?:\s[A-Z]+)*)\s+(PRK\.?\s+.+)', re.IGNORECASE)
    # Regex to capture general voter information with or without prefix, including commas
    voter_info_regex = re.compile(r'(\d+)?\s*[*]?\s*([A-Z]+(?:\s[A-Z]+)*),?\s+([A-Z]+(?:\s[A-Z]+)*)\s+([A-Z]+(?:\s[A-Z]+)*)\s+(PRK\.?\s+.+)', re.IGNORECASE)
    # Regex to capture lines starting with a number
    number_start_regex = re.compile(r'^\d+\s+')

    final_add = {
        'PRK.',
        'CENTRO',
        'PUROK',
        'RK',
        'RK.',
        'PRK.1',
        'PRPK.'
    }

    def convert_line_to_list(line):
        # Define the pattern to split the line based on commas
        parts = [part.strip() for part in line.split(',')]
        return parts

    def seprate_number_name(numberName):
        onlyNumber = re.findall(r'\b\d+\b', numberName)
        return onlyNumber

    def remove_number(number_name, only_numbers):
        # Remove all found numbers from the input string
        for num in only_numbers:
            number_name = number_name.replace(num, "").strip()
        return number_name

    def extract_only_number_firstName(data):
        onlyNumber = seprate_number_name(data)
        final = remove_number(data, onlyNumber)
        firstName = ""
        remove_special = final.split(" ")
        if len(remove_special) > 1:
            firstName = remove_special[1]
        else:
            firstName = remove_special[0]
        return onlyNumber, firstName

    def split_address(address):
        parts = [part.strip() for part in address.split(',')]
        final_parts = []
        for part in parts:
            sub_parts = [sub_part for sub_part in part.split() if sub_part]
            final_parts.extend(sub_parts)
        return final_parts

    def split_array(arr, final_add):
        final_add_set = set(final_add)
        split_index = None
        for i, item in enumerate(arr):
            if item in final_add_set:
                split_index = i
                break
        if split_index is None:
            return [arr, []]
        first_part = arr[:split_index]
        second_part = arr[split_index:]
        return [first_part, second_part]

    # Function to extract voter information from a line
    def extract_voter_info(line, barangay):
        match = voter_info_regex.match(line)
        if match:
            voter_no = match.group(1) if match.group(1) else ""
            last_name = match.group(2)
            first_name = match.group(3)
            middle_name = match.group(4)
            address = match.group(5)
            full_name = f"{last_name} {first_name} {middle_name}".strip()
            return [voter_no, full_name, address, barangay]
        return None

    def get_non_matching(result):
        # Initialize variables
        final_address = []
        main_data = []

        # Convert the result to a formatted list
        formatted_list = convert_line_to_list(result)

        # Extract relevant data
        precint_no = formatted_list[0]
        onlyNumber, firstName = extract_only_number_firstName(formatted_list[1])

        # Split the address
        get_address = split_address(formatted_list[2])

        # Determine how to handle the address
        if len(get_address) == 1:
            final_address = formatted_list[2] + " " + formatted_list[3]
            final_address = final_address.split(" ")
        else:
            final_address = get_address

        # Separate address parts
        seprating_address = split_array(final_address, final_add)

        # Extract middle name/last name and address info
        middle_name_last_name = seprating_address[0]
        address_info = seprating_address[1]

        # Concatenate first name with middle name/last name
        fullname = " ".join([firstName] + middle_name_last_name)
        address_data = " ".join(address_info)
        
        return onlyNumber[0], fullname, address_data

    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        city = province = barangay = ""
        current_precinct = ""
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split("\n")
            for line in lines:
                # Check and extract precinct number
                prec_match = prec_regex.search(line)
                if prec_match:
                    current_precinct = prec_match.group(1)
                    if current_precinct not in precinct_voters:
                        precinct_voters[current_precinct] = []

                # Check and extract city, province, and barangay
                city_match = city_regex.search(line)
                if city_match:
                    city = city_match.group(1)
                province_match = province_regex.search(line)
                if province_match:
                    province = province_match.group(1)
                barangay_match = barangay_line_regex.search(line)
                if barangay_match:
                    barangay = barangay_match.group(1)

                # Extract voter information and store it under the current precinct
                voter_info = extract_voter_info(line, barangay)
                if voter_info:
                    # Add city and province to the voter info
                    voter_info.extend([city, province])
                    precinct_voters[current_precinct].append(voter_info)
                else:
                    # Check if the line starts with a number and does not match the primary format
                    if number_start_regex.match(line):
                        formatted_result = f"{current_precinct}, {line}, {barangay}, {city}, {province}"
                        onlyNumber, fullname, address_info = get_non_matching(formatted_result)
                        voter_info = [onlyNumber, fullname, address_info, barangay, city, province]
                        precinct_voters[current_precinct].append(voter_info)

                # Extract names with an asterisk followed by a space and store them
                asterisk_name_match = asterisk_name_regex.match(line)
                if asterisk_name_match:
                    last_name = asterisk_name_match.group(1)
                    first_name = asterisk_name_match.group(2)
                    middle_name = asterisk_name_match.group(3)
                    address = asterisk_name_match.group(4)
                    full_name = f"{last_name} {first_name} {middle_name}".strip()
                    voter_info = ['', full_name, address, barangay]  
                    voter_info.extend([city, province])
                    if current_precinct not in precinct_voters:
                        precinct_voters[current_precinct] = []
                    precinct_voters[current_precinct].append(voter_info)

    # Process and store the extracted data in the results dictionary
    for precinct, voters in precinct_voters.items():
        results[precinct] = {
            "precinct": precinct,
            "total_voters": len(voters),
            "voters": [dict(zip(["Voter No", "Full Name", "Address", "Barangay", "City", "Province"], voter)) for voter in voters]
        }

    return results


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.post("/addusers")
async def create_users(file: UploadFile = File(...)):
    try:
        # Check for valid file type
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")
        print("filename", file.filename)
        # Secure the filename and save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        
        # Save the uploaded file locally
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract voter information from the uploaded PDF
        voter_information = extract_voter_information_from_pdf(file_path)

        # Iterate through each precinct and add to Firestore
        for precinct, data in voter_information.items():
            print(f"Processing precinct: {precinct}")

            # Reference to the precinct document
            precinct_ref = db.collection(VOTERS_COLLECTION).document(precinct)
            precinct_doc = precinct_ref.get()

            # Check if precinct document already exists
            if precinct_doc.exists:
                return JSONResponse(
                    content={"error": f"Precinct '{precinct}' already exists."}, 
                    status_code=400
                )

            # Add new precinct and its voters
            precinct_ref.set({"total_voters": data['total_voters']})
            voters_ref = precinct_ref.collection('voters')

            # Add each voter to the voters sub-collection
            for voter in data["voters"]:
                voters_ref.add(voter)

        return JSONResponse(content={"message": "Data added successfully"}, status_code=201)
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the file")

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import firebase_admin
from firebase_admin import credentials, firestore
import os
from datetime import datetime
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import logging
import pdfplumber
import re
import nltk
from typing import Optional
from werkzeug.utils import secure_filename
app = FastAPI()

# Firebase initialization
cred = credentials.Certificate("vels-d0547-firebase-adminsdk-6kni4-2d374ae1f1.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
USERS_COLLECTION = "users"
VOTERS_COLLECTION ="Voters"
ELECTION_COLLECTION ="Election"
ALLOCATE_COLLECTION ="allocate"
SURVEY_COLLECTION ="Survey"
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
    candidateId: Optional[str] = None 

class ElectionData(BaseModel):
    title: str
    electionId: str
    surveyorStartDate: str
    surveyorEndDate: str
    resultDate: str
class AllocateData(BaseModel):
    electionId: str
    surveyorId: str
    precintList: str

class SurveyDetails(BaseModel):
    electionId: str
    surveyorId: str
    precintList: str
    userDocumentId: str
    gender: str
    age: str
    dob: str
    civil_status: str
    code1: str
    tag1: str
    code2: str
    tag2: str
    code3: str
    tag3: str
    code4: str
    tag4: str
    candidateId: str
    remarks: str
    


@app.post('/signin')
def signin(data: SigninData):
    # Extract data from the request
    email = data.email
    password = data.password
    print(f"User signing in: {email}")
    
    if not email or not password:
        return JSONResponse(content={"error": "Missing data"}, status_code=400)
    
    # Fetch user from Firestore
    users_ref = db.collection(USERS_COLLECTION)
    query = users_ref.where('email', '==', email).limit(1).get()

    if not query:
        return JSONResponse(content={"error": "User not found"}, status_code=404)
    
    user_doc = query[0].to_dict()
    
    # Check if the provided password matches the stored password
    if user_doc.get('password') == password:
        # Return only the username and ID
        user_response = {
            "id": query[0].id,  # Firestore document ID
            "username": user_doc.get('username'),
            "selectedMode": user_doc.get('selectedMode')
        }
        return JSONResponse(content={"message": "Sign in successful", "user": user_response}, status_code=200)
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
    candidateId = data.candidateId

    if not username or not email or not password:
        raise HTTPException(status_code=400, detail="Missing data")

    # Prepare user data dictionary
    user_data = {
        'username': username,
        'email': email,
        'password': password,
        'contact': contact,
        'selectedMode': selectedMode,
    }

    # Add candidateId if it exists
    if candidateId:
        user_data['candidateId'] = candidateId

    # Store the new user in the database
    users_ref = db.collection(USERS_COLLECTION)  
    new_user_ref = users_ref.document() 
    new_user_ref.set(user_data)

    # Return success message
    return JSONResponse(content={"message": "Signup successful"}, status_code=201)


@app.get('/users')
def get_all_users():
    try:
        users_ref = db.collection(USERS_COLLECTION)  
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


@app.get('/precints')
def get_all_precints():
    try:
        # Fetch all documents from the collection
        users_ref = db.collection(VOTERS_COLLECTION)
        docs = users_ref.stream()
        logging.info(f"users_ref: {users_ref}")
        # Extract document IDs
        doc_ids = [doc.id for doc in docs]

        return JSONResponse(content={"doc_ids": doc_ids}, status_code=200)
    
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Failed to retrieve document IDs"}, status_code=500)

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




@app.post('/addElection')
def addElection(data: ElectionData):
    # Extract data from the request
    title = data.title
    electionId = data.electionId
    surveyorStartDate = data.surveyorStartDate
    surveyorEndDate = data.surveyorEndDate
    resultDate = data.resultDate
    created_at = datetime.now().isoformat()

    if not title or not electionId or not surveyorStartDate:
        return JSONResponse(content={"error": "Missing data"}, status_code=400)
    
    election_ref = db.collection(ELECTION_COLLECTION)  
    new_election_ref = election_ref.document()  
    new_election_ref.set({
        'electionId': title,
        'electionName': electionId,
        'surveyorStartDate': surveyorStartDate,
        'surveyorEndDate': surveyorEndDate,
        'resultDate': resultDate,
        'created_at':created_at,
        'isAllocated':False
    })
    logging.info(f"new_election_ref: {new_election_ref}")
    # Return success message
    return JSONResponse(content={"message": "Election Added successful"}, status_code=201)




@app.get('/allelection')
def allelection():
    try:
        # Fetch all documents from the collection
        users_ref = db.collection(ELECTION_COLLECTION)
        docs = users_ref.stream()
        logging.info(f"users_ref: {users_ref}")

        # Extract document data
        all_elections = []
        for doc in docs:
            election_data = doc.to_dict()  # Get document data as a dictionary
            election_data["id"] = doc.id    # Add document ID to the data
            all_elections.append(election_data)

        return JSONResponse(content=all_elections, status_code=200)
    
    except Exception as e:
        logging.error(f"Error: {e}")
        return JSONResponse(content={"error": "Failed to retrieve election data"}, status_code=500)


@app.get('/get_userDetails/{userid}')
def get_userDetails(userid: str):
    try:
        users_ref = db.collection(USERS_COLLECTION).document(userid)
        user_doc = users_ref.get()  # Fetch the document

        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User not found")

        user_data = user_doc.to_dict()
        return user_data
     
    
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Failed to retrieve users"}, status_code=500)
def get_userName(userid: str):
    try:
        users_ref = db.collection(USERS_COLLECTION).document(userid)
        user_doc = users_ref.get()  # Fetch the document

        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User not found")

        # Convert document data to a dictionary
        user_data = user_doc.to_dict()
        return user_data['username']
     
    
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Failed to retrieve users"}, status_code=500)


@app.get('/get_surveyor')
def get_surveyor():
    try:
        users_ref = db.collection(USERS_COLLECTION)
        docs = users_ref.stream()

        users = []
        for doc in docs:
            user_data = doc.to_dict()
            user_data['id'] = doc.id  # This is the user ID
            
            # Check if the user is a surveyor
            if user_data.get('selectedMode').lower() == 'surveyor':
                # Only append the 'id' and 'username' fields
                users.append({
                    'userId': user_data['id'],
                    'userName': user_data.get('username')  # Ensure 'username' exists in the data
                })

        return JSONResponse(content=users, status_code=200)
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Failed to retrieve users"}, status_code=500)


@app.get('/get_allocation_list')
def get_allocation_list():
    try:
        users_ref = db.collection(ALLOCATE_COLLECTION)  
        docs = users_ref.stream()

        users = []
        for doc in docs:
            user_data = doc.to_dict()
            user_data['id'] = doc.id  
            username = get_userName(user_data['surveyorId'])
            users.append({
                'userId': user_data['id'],
                'surveyorName': username,
                'precintList': user_data['precintList'],
                'electionId': user_data['electionId'],
                'created_at': user_data['created_at'],
            })

        return JSONResponse(content=users, status_code=200)
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Failed to retrieve users"}, status_code=500)

@app.get('/get_election_list')
def get_election_list():
    try:
        users_ref = db.collection(ELECTION_COLLECTION)  
        docs = users_ref.stream()

        users = []
        for doc in docs:
            user_data = doc.to_dict()
            user_data['id'] = doc.id 
            if user_data['isAllocated'] == False:
                users.append({
                    'id': user_data['id'],
                    'electionId': user_data.get('electionId') ,
                    'electionName': user_data.get('electionName') 
                })
            else: 
                return JSONResponse(content=users, status_code=200)
        return JSONResponse(content=users, status_code=200)
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Failed to retrieve users"}, status_code=500)


@app.post('/allocateInfo')
def allocateInfo(data: AllocateData):
    # Extract data from the request
    electionId = data.electionId
    surveyorId = data.surveyorId
    precintList = data.precintList
    created_at = datetime.now().isoformat()

    
    users_ref = db.collection(ALLOCATE_COLLECTION)  
    new_user_ref = users_ref.document()  
    new_user_ref.set({
        'electionId': electionId,
        'surveyorId': surveyorId,
        'precintList': precintList,
        'created_at': created_at
    })
    
    # Return success message
    return JSONResponse(content={"message": "Allocated Successfully"}, status_code=201)

@app.get('/get_candidate')
def get_candidate():
    try:
        users_ref = db.collection(USERS_COLLECTION)
        docs = users_ref.stream()

        users = []
        for doc in docs:
            user_data = doc.to_dict()
            user_data['id'] = doc.id  # This is the user ID
            
            # Check if the user is a surveyor
            if user_data.get('selectedMode').lower() == 'candidate':
                # Only append the 'id' and 'username' fields
                users.append({
                    'userId': user_data['id'],
                    'userName': user_data.get('username')  
                })

        return JSONResponse(content=users, status_code=200)
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Failed to retrieve users"}, status_code=500)

@app.get('/get_allocated_list')
def get_allocated_list():
    try:
        users_ref = db.collection(ALLOCATE_COLLECTION)
        docs = users_ref.stream()

        users = []
        for doc in docs:
            user_data = doc.to_dict()
            user_data['id'] = doc.id  # This is the user ID
            users.append(user_data)
          

        return JSONResponse(content=users, status_code=200)
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Failed to retrieve users"}, status_code=500)


@app.get('/verifierCheck/{verifierId}/{documentId}')
def verifierCheck(verifierId: str, documentId: str):
    try:
        # Reference to the document
        users_ref = db.collection(ALLOCATE_COLLECTION).document(documentId)
        
        # Retrieve the document
        doc = users_ref.get()

        if not doc.exists:
            return JSONResponse(content={"error": "Document not found"}, status_code=404)


        users_ref.update({"verifierId": verifierId,"isOpen": True})

        return JSONResponse(status_code=200, content={"message": "Verifier ID added successfully"})
    
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Failed to update the document"}, status_code=500)


def getElectionDetails(electionId):
    try:
        # Query documents where electionId matches the given electionId
        docs = db.collection(ELECTION_COLLECTION).where('electionId', '==', electionId).stream()
        
        # List to store election details
        election_details = []

        for doc in docs:
            election_details.append(doc.to_dict())  # Convert each document to a dictionary and add to the list

        # Return the election details or None if not found
        return election_details if election_details else None
    except Exception as e:
        print(f"Error: {e}")
        return None  # Return None in case of an error
def getVerifierName(verifierId):
    try:
        # Get the document by verifierId
        doc_ref = db.collection(USERS_COLLECTION).document(verifierId)  # Use the correct reference
        doc = doc_ref.get()  # Call get() on the document reference
        
        # Check if the document exists
        if doc.exists:
            return doc.to_dict().get('username')  # Return the 'username' field of the document
        else:
            return None  # Document not found
    except Exception as e:
        print(f"Error: {e}")
        return None  # Return None in case of an error



@app.get('/getVerifiedSurveyDetails/{surveyorId}/')
def getVerifiedSurveyDetails(surveyorId: str):
    try:
        docs = db.collection(ALLOCATE_COLLECTION).where('isOpen', '==', True).where('surveyorId', '==', surveyorId).stream()

        # List to store the details of documents
        open_documents = []

        for doc in docs:
            doc_data = doc.to_dict()  # Convert document to dictionary
            electionId = doc_data.get('electionId')
            verifierId = doc_data.get('verifierId')

            # Retrieve election details
            election_details = getElectionDetails(electionId) or {}
            verifier_name = getVerifierName(verifierId) or "Unknown Verifier"

            # Combine the document data with election details and verifier name
            combined_data = {
                **doc_data,
                "electionDetails": election_details,
                "verifierName": verifier_name
            }
            open_documents.append(combined_data)  # Append the combined data to the list

        # Return the combined list of open documents with details
        if open_documents:
            return JSONResponse(status_code=200, content={"openDocuments": open_documents})
        else:
            return JSONResponse(status_code=404, content={"message": "No open documents found for the given surveyor ID."})

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Failed to retrieve documents"}, status_code=500)

@app.post("/surveys/")
async def create_survey(survey: SurveyDetails):
    # Prepare the data to be added to Firestore
    survey_data = survey.dict()
    survey_data['created_at'] = datetime.utcnow().isoformat()  # Add created_at timestamp

    # Add the survey data to the Firestore collection
    try:
        db.collection(SURVEY_COLLECTION).add(survey_data)
        return JSONResponse(content={"message": "Survey created successfully"}, status_code=201)
    except Exception as e:
        # Include the exception message and code in the response
        return JSONResponse(
            content={"detail": str(e), "code": 500},
            status_code=500
        )


logging.basicConfig(level=logging.INFO)

@app.get('/surveys/{surveyorId}/{electionId}/{precintNo}/')
def getSurveyById(surveyorId: str,electionId: str,precintNo: str):
    try:
        # Check if surveyorId is valid
        if not surveyorId:
            raise HTTPException(status_code=400, detail="Invalid surveyor ID.")

        # Fetch documents from Firestore using keyword arguments
        docs = db.collection(SURVEY_COLLECTION).where('surveyorId', '==', surveyorId).where('electionId', '==', electionId).where('precintList', '==', precintNo).stream()

        # Convert documents to a list of dictionaries
        survey_data = []
        for doc in docs:
            doc_dict = doc.to_dict()
            print(doc_dict)
            if 'created_at' in doc_dict:
                created_at = doc_dict['created_at']
                print(created_at)
            doc_dict['created_at'] = created_at.isoformat()
            survey_data.append(doc_dict)

        if not survey_data:
            # If no surveys found, return 404
            raise HTTPException(status_code=404, detail="No surveys found for this surveyor ID.")

        # Return as JSON response with 200 OK status
        return JSONResponse(content=survey_data, status_code=200)

    except Exception as e:
        logging.error(f"Error occurred: {e}")  # Log the error
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get('/excludeUserId/{surveyorId}/{electionId}/{precintNo}/')
def getexcludeUserId(surveyorId: str,electionId: str,precintNo: str):
    try:
        # Check if surveyorId is valid
        if not surveyorId:
            raise HTTPException(status_code=400, detail="Invalid surveyor ID.")

        # Fetch documents from Firestore using keyword arguments
        docs = db.collection(SURVEY_COLLECTION).where('surveyorId', '==', surveyorId).where('electionId', '==', electionId).where('precintList', '==', precintNo).stream()

        # Convert documents to a list of dictionaries
        excluded_list = []
        for doc in docs:
            doc_dict = doc.to_dict()
            excluded_list.append(doc_dict['userDocumentId'])

        if not excluded_list:
            # If no surveys found, return 404
            raise HTTPException(status_code=404, detail="No surveys found for this surveyor ID.")

        # Return as JSON response with 200 OK status
        return JSONResponse(content=excluded_list, status_code=200)

    except Exception as e:
        logging.error(f"Error occurred: {e}")  # Log the error
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    
#get the updated details
 
@app.get('/surveyData/{userDocumentId}')
def surveyData(userDocumentId: str):
    try:
        docs = db.collection(SURVEY_COLLECTION).where('userDocumentId', '==', userDocumentId).stream()
        excluded_list = []
        
        for doc in docs:
            doc_dict = doc.to_dict()  # Convert each document to a dictionary
            
            # Check if 'created_at' is in the document
            if 'created_at' in doc_dict:
                created_at = doc_dict['created_at']
                username = get_userName(doc_dict['candidateId'])
                
                # Format the datetime to a readable string
                formatted_datetime = created_at.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
                doc_dict['created_at'] = formatted_datetime  # Update the dictionary with the formatted datetime
                doc_dict['candidate_name'] = username  # Add username to the dictionary
            
            excluded_list.append(doc_dict)  # Append the document to the list
        print(excluded_list)
        return JSONResponse(content=excluded_list, status_code=200)

    except Exception as e:
        print(f"Error retrieving survey data: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving survey data.")
        

        
        
from datetime import datetime
from datetime import datetime, timedelta
from collections import defaultdict
from dateutil import parser
@app.get('/surveyElectionData')
async def surveyElectionData():
    try:
        # Get all documents in the collection
        docs = db.collection(SURVEY_COLLECTION).stream()
        
        # Convert documents to a list of dictionaries with formatted datetime
        survey_data = []
        for doc in docs:
            doc_dict = doc.to_dict()
            if 'created_at' in doc_dict and isinstance(doc_dict['created_at'], datetime):
                formatted_datetime = doc_dict['created_at'].strftime('%Y-%m-%d %H:%M:%S.%f %Z')
                doc_dict['created_at'] = formatted_datetime
            survey_data.append(doc_dict)

        # Calculate candidate vote counts
        candidate_votes = defaultdict(int)
        for entry in survey_data:
            candidate_votes[entry['candidateId']] += 1

        # Calculate weekly votes
        weekly_votes = defaultdict(lambda: defaultdict(int))
        for entry in survey_data:
            created_at_str = entry['created_at']
            
            # Clean the string to handle different formats
            if 'UTC' in created_at_str:
                created_at_str = created_at_str.replace(' UTC', '')  # Remove ' UTC'
            created_at = parser.isoparse(created_at_str)

            week_start = created_at - timedelta(days=created_at.weekday())  # Get start of the week
            week_end = week_start + timedelta(days=6)  # Calculate end of the week
            week_str = week_start.strftime('%Y-%m-%d')
            candidate_id = entry['candidateId']

            # Increment the count for the specific candidate in that week
            weekly_votes[week_str][candidate_id] += 1
       
        # Format results for weekly votes
        weekly_votes_list = [
            {
                "week_start": week_start.strftime('%Y-%m-%d'),
                "week_end": week_end.strftime('%Y-%m-%d'),
                "candidates": [
                    {"candidateId": candidate, "votes": count,"username": get_userName(candidate)} for candidate, count in candidates.items()
                ]
            }
            for week_str, candidates in weekly_votes.items()
            for week_start in [parser.isoparse(week_str)]  # Convert week_str back to datetime
            for week_end in [week_start + timedelta(days=6)]  # Calculate week end
        ]
       
        # Format results for candidate votes
        candidate_votes_list = [{"candidateId": candidate,"username": get_userName(candidate), "votes": count} for candidate, count in candidate_votes.items()]

        # Return the response as JSON
        return JSONResponse(content={
          
            "candidate_votes": candidate_votes_list,
            "weekly_votes": weekly_votes_list
        }, status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/get_assistantDetails/{userid}')
def get_assistantDetails(userid: str):
    try:
        users_ref = db.collection(USERS_COLLECTION).where('candidateId', '==', userid).stream()

        # Since `stream()` returns a generator, iterate over it
        user_data = None
        user_id = None
        for user_doc in users_ref:
            if user_doc.exists:
                user_data = user_doc.to_dict()
                user_id = user_doc.id  # Get document ID
                break  # Exit after finding the first match

        if user_data is None:
            raise HTTPException(status_code=404, detail="User not found")

        # Return both the document ID and data
        return {"id": user_id, "data": user_data}

    except Exception as e:
        print(f"Error retrieving survey data: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while retrieving get_assistantDetails data.")

@app.get('/assistant_request/{userid}/{status}')
def get_assistantDetails(userid: str, status: str):
    try:
        # Reference the document for the given userid
        user_ref = db.collection(USERS_COLLECTION).document(userid)

        # Update the document by adding or updating the 'status' field
        user_ref.update({'status': status})

        # Retrieve the updated user document
        updated_user = user_ref.get().to_dict()

        # Return a success message and updated user data
        return JSONResponse(content={"message": "Success"}, status_code=200)

    except Exception as e:
        print(f"Error retrieving or updating assistant details: {e}")
        # Raise HTTP 500 error if something goes wrong
        raise HTTPException(status_code=500, detail="An error occurred while retrieving get_assistantDetails data.")
    

@app.get('/getGraphDetails/')
def getGraphDetails():
    try:
        # Fetch all documents from the SURVEY_COLLECTION
        survey_ref = db.collection(SURVEY_COLLECTION).stream()
        survey_data = []
        
        # Loop through each document in the collection
        for survey in survey_ref:
            survey_dict = survey.to_dict()
            
            # Extract necessary information
            userdocumentId = survey_dict.get('userDocumentId')
            candidateId = survey_dict.get('candidateId')
            electionId = survey_dict.get('electionId')
            precintList = survey_dict.get('precintList')
            
            # Fetch voter data
            voter_ref = db.collection(VOTERS_COLLECTION).document(precintList).collection('voters').document(userdocumentId)
            voter_snapshot = voter_ref.get()
            
            if voter_snapshot.exists:
                voter_data = voter_snapshot.to_dict()
                
                # Append data including voting information
                survey_data.append({
                    "userdocumentId": userdocumentId,
                    "candidateId": candidateId,
                    "electionId": electionId,
                    "voterData": voter_data
                })

        # Transform data for Power BI
        transformed_data = {}
        
        for entry in survey_data:
            # Fetch candidate username
            candidate_ref = db.collection(USERS_COLLECTION).document(entry["candidateId"]).get()
            candidate_username = candidate_ref.to_dict().get("username", "Unknown") if candidate_ref.exists else "Unknown"

            # Use candidateId, electionId, Province, City, and precintList to create a unique key
            key = (entry["candidateId"], entry["electionId"], entry["voterData"]["Province"], entry["voterData"]["City"], precintList)
            if key not in transformed_data:
                transformed_data[key] = {
                    "vote_count": 0,
                    "candidate_username": candidate_username
                }
            transformed_data[key]["vote_count"] += 1  # Count votes

        # Prepare the final response
        final_response = []
        for (candidateId, electionId, province, city, precintList), data in transformed_data.items():
            final_response.append({
                "candidateId": candidateId,
                "electionId": electionId,
                "Province": province,
                "City": city,
                "VoteCount": data["vote_count"],
                "CandidateUsername": data["candidate_username"],
                "PrecintList": precintList
            })

        # Return the transformed data
        return JSONResponse(content=final_response)

    except Exception as e:
        logging.error(f"Error retrieving surveyor details: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving surveyor details")


from datetime import datetime, timedelta
from collections import defaultdict
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from google.cloud.firestore import DocumentSnapshot

@app.get('/getWeeklyReportprecint/')
def getWeeklyReportprecint():
    try:
        # Fetch all documents from the SURVEY_COLLECTION
        survey_ref = db.collection(SURVEY_COLLECTION).stream()
        survey_data = []
        
        # Loop through each document in the collection
        for survey in survey_ref:
            survey_dict = survey.to_dict()
            
            # Extract necessary information
            userdocumentId = survey_dict.get('userDocumentId')
            candidateId = survey_dict.get('candidateId')
            created_at = survey_dict.get('created_at')  # This is likely a Timestamp object
            precintList = survey_dict.get('precintList')
            
            # Print the created_at for debugging
            print(f"Raw created_at: {created_at}, Type: {type(created_at)}")  # Log the raw value and type
            
            # Convert Firestore Timestamp to datetime object
            if isinstance(created_at, DocumentSnapshot):
                created_at_date = created_at.to_datetime()  # Use the Firestore Timestamp method
                print(f"Converted created_at (from DocumentSnapshot): {created_at_date}")  # Log converted value
            elif isinstance(created_at, datetime):
                created_at_date = created_at  # If it's already a datetime
                print(f"created_at is already a datetime: {created_at_date}")  # Log if it's already a datetime
            else:
                created_at_date = datetime.fromisoformat(created_at[:-1]) if isinstance(created_at, str) else None
                print(f"created_at after fallback: {created_at_date}")  # Log fallback value

            if created_at_date is None:
                logging.error(f"created_at could not be converted for userDocumentId: {userdocumentId}")
                continue  # Skip this entry if created_at is invalid
            
            # Append data including voting information
            survey_data.append({
                "userdocumentId": userdocumentId,
                "candidateId": candidateId,
                "created_at": created_at_date,
                "precintList": precintList
            })

        # Group votes by week and candidate
        weekly_votes = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'username': None, 'precintList': None}))
        # {week: {candidateId: {'count': vote_count, 'username': username, 'precintList': precintList}}}
        
        for entry in survey_data:
            # Determine the start of the week (Monday)
            week_start = entry["created_at"] - timedelta(days=entry["created_at"].weekday())
            week_str = week_start.strftime("%Y-W%W")  # ISO format for week
            week_display = f"Week {week_start.isocalendar()[1]} of {week_start.year}"  # More readable format

            # Count the vote for the candidate
            candidate_id = entry["candidateId"]
            weekly_votes[week_display][candidate_id]['count'] += 1
            
            # Fetch candidate username
            candidate_ref = db.collection(USERS_COLLECTION).document(candidate_id).get()
            candidate_username = candidate_ref.to_dict().get("username", "Unknown") if candidate_ref.exists else "Unknown"
            weekly_votes[week_display][candidate_id]['username'] = candidate_username
            
            # Store the precintList
            if weekly_votes[week_display][candidate_id]['precintList'] is None:
                weekly_votes[week_display][candidate_id]['precintList'] = entry["precintList"]
            else:
                # If there are multiple precincts, you could choose to store them as a list or a string.
                # Here, we'll keep it simple by ensuring it's the same if not already set.
                weekly_votes[week_display][candidate_id]['precintList'] = entry["precintList"]

        # Prepare the final response
        final_response = []
        for week, candidates in weekly_votes.items():
            for candidateId, data in candidates.items():
                final_response.append({
                    "Week": week,
                    "CandidateId": candidateId,
                    "CandidateUsername": data['username'],
                    "VoteCount": data['count'],
                    "PrecintList": data['precintList']  # Include the precintList
                })

        # Return the transformed data
        return JSONResponse(content=final_response)

    except Exception as e:
        logging.error(f"Error retrieving weekly report: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving weekly report")



@app.get('/leader_graph_details/')
def leader_graph_details():
    try:
        # Fetch all documents from the SURVEY_COLLECTION
        survey_ref = db.collection(SURVEY_COLLECTION).stream()
        survey_data = []
        
        # Loop through each document in the collection
        for survey in survey_ref:
            survey_dict = survey.to_dict()
            
            # Extract necessary information
            userdocumentId = survey_dict.get('userDocumentId')
            userage = survey_dict.get('age')
            user_gender = survey_dict.get('gender')
            candidateId = survey_dict.get('candidateId')
            electionId = survey_dict.get('electionId')
            precintList = survey_dict.get('precintList')
            
            # Fetch voter data
            voter_ref = db.collection(VOTERS_COLLECTION).document(precintList).collection('voters').document(userdocumentId)
            voter_snapshot = voter_ref.get()
            
            if voter_snapshot.exists:
                voter_data = voter_snapshot.to_dict()
                
                # Append data including voting information
                survey_data.append({
                    "userdocumentId": userdocumentId,
                    "candidateId": candidateId,
                    "electionId": electionId,
                    "voterData": voter_data,
                    "userage": userage,
                    "user_gender": user_gender  # Include user gender
                })

        # Transform data for Power BI
        transformed_data = {}
        
        for entry in survey_data:
            # Fetch candidate username
            candidate_ref = db.collection(USERS_COLLECTION).document(entry["candidateId"]).get()
            candidate_username = candidate_ref.to_dict().get("username", "Unknown") if candidate_ref.exists else "Unknown"

            # Use candidateId, electionId, Province, City, and precintList to create a unique key
            key = (entry["candidateId"], entry["electionId"], entry["voterData"]["Province"], entry["voterData"]["City"], precintList)
            if key not in transformed_data:
                transformed_data[key] = {
                    "vote_count": 0,
                    "candidate_username": candidate_username,
                    "userages": [],  # Store user ages for this key
                    "user_genders": []  # Store user genders for this key
                }
            transformed_data[key]["vote_count"] += 1  # Count votes
            transformed_data[key]["userages"].append(entry["userage"])  # Collect user ages
            transformed_data[key]["user_genders"].append(entry["user_gender"])  # Collect user genders

        # Prepare the final response
        final_response = []
        for (candidateId, electionId, province, city, precintList), data in transformed_data.items():
            final_response.append({
                "candidateId": candidateId,
                "electionId": electionId,
                "Province": province,
                "City": city,
                "VoteCount": data["vote_count"],
                "CandidateUsername": data["candidate_username"],
                "PrecintList": precintList,
                "UserAges": data["userages"],  # Include user ages
                "UserGenders": data["user_genders"]  # Include user genders
            })

        # Return the transformed data
        return JSONResponse(content=final_response)

    except Exception as e:
        logging.error(f"Error retrieving surveyor details: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving surveyor details")


#uvicorn app_api:app --host 0.0.0.0 --port 8000 --proxy-headers



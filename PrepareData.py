import pandas as pd
import numpy as np

# Prepare data for gender recognition
def prepare_data_CNN(csv_path:str, num_exp: int, num_train: int, num_test: int, action: bool = True):
    # Load the data from csv metadata file
    df = pd.read_csv(csv_path)
    # Create a data structure to store the images' name and the corresponding label
    data_structure = {}
    
    print("Preparing CNN Data\n")

    # Populate the data structure
    for indExp in range(num_exp):
        print(f"\tExp {indExp}")
        data_structure[indExp] = {}
        df['check'] = False
        data_structure[indExp]['train'], df = prepare_data_CNN_train(num_train= num_train, df = df)

        if action:
            data_structure[indExp]['test']= prepare_data_CNN_test(num_test= num_test, df=df) 
        else:
            temp_set = set(data_structure[indExp]['train']['labels_id'])
            data_structure[indExp]['test']= prepare_data_CNN_test(num_test= num_test, df=df, extracted_person_id_list=list(temp_set)) 
    
    print("CNN Data Preparation Completed\n")
    return data_structure

def prepare_data_CNN_train(num_train: int,  df: pd.DataFrame):
    result_dict = {
                "labels": [],
                "labels_id": [],
                "images": []
            }
    
    gender = ['male',  'female']

    print("\t\tTraining")

    for gend in gender:
        # Extract the person id without accessories
        person_id_list = df.loc[(df['gender'] == gend), 'id'].unique()
        for _ in range(num_train):
            for i in range(0, len(person_id_list)):
                # Extract a person id
                person_id = np.random.choice(person_id_list)

                '''
                    Exclude people who no longer have palm and back images to extract
                '''
                if (len(df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar'))&(df['accessories'] == 0)]) == 0 or len(df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal'))&(df['accessories'] == 0)]) == 0
                        ) or (
                    df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar'))&(df['accessories'] == 0), 'check'].all() or df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal'))&(df['accessories'] == 0), 'check'].all()): 
                  
                    person_id_list = np.delete(person_id_list, np.where(person_id_list == person_id)[0])
                    continue 
                else:
                    break
           
            '''
            Filter by palm/back side
            In the training dataset we exclude images with obstructions (accessories) -> to avoid bias
            Finally we take the name of the image
            With .sample we extract # num_train or num_test elements from the dataset and with replace=False we avoid extracting duplicates
            '''
            result_dict["labels"].append(0 if df.loc[df["id"] == person_id,'gender'].iloc[0] == "male" else 1)
            result_dict["labels_id"].append(person_id)
            '''
            From the entire df dataframe
            we filter on the id of a single person
            I take the palms or backs
            We randomly choose a palm and a hand
            With check == True the image is excluded because it has already been taken
            '''  
            palmar_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("palmar"))&(df['accessories'] == 0)&(df["check"] == False),'imageName'].sample(n=1, replace=False).to_list()
            dorsal_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("dorsal"))&(df['accessories'] == 0)&(df["check"] == False),'imageName'].sample(n=1, replace=False).to_list()
            
            '''
            The check field indicates that an image has already been taken and therefore cannot be retrieved.
            '''
            df.loc[(df["imageName"] == palmar_img[0]),'check'] = True
            df.loc[(df["imageName"] == dorsal_img[0]),'check'] = True

            result_dict["images"].append([palmar_img, dorsal_img])
    return result_dict, df

def prepare_data_CNN_test(num_test: int, df: pd.DataFrame, extracted_person_id_list: np.array = None):
    result_dict = {
        "labels": [],
        "labels_id": [],
        "images": []
    } 
    
    male_female_list = ['male', 'female']
    person_id_list = []

    print("\t\tTesting\n")

    for gender in male_female_list:
        if extracted_person_id_list is None:
            person_id_list = df.loc[(df['gender'] == gender), 'id'].unique()
        else:
            person_id_list = extracted_person_id_list

        for person_id in person_id_list:
            if df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar')), 'check'].all() or df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal')), 'check'].all():
                person_id_list = np.delete(person_id_list, np.where(person_id_list == person_id)[0])

        for _ in range(num_test):
            person_id = np.random.choice(person_id_list)
            '''
            Filter by palm/back side
            In the training dataset we exclude images with obstructions (accessories) -> to avoid bias
            Finally we take the name of the image
            With .sample we extract # num_train or num_test elements from the dataset and with replace=False it avoids extracting duplicates
            '''
            result_dict["labels"].append(0 if df.loc[df["id"] == person_id,'gender'].iloc[0] == "male" else 1)
            result_dict["labels_id"].append(person_id)
            '''
            From the entire dataframe df
            we filter on a single person id
            I take the palms or backs
            We randomly choose a palm and a hand
            '''

            palmar_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("palmar"))&(df["check"] == False),'imageName'].sample(n=1, replace=False).to_list()
            dorsal_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("dorsal"))&(df["check"] == False),'imageName'].sample(n=1, replace=False).to_list()
           
            '''
            The check field indicates that an image has already been taken and therefore cannot be retrieved.            
            '''
            df.loc[(df["imageName"] == palmar_img[0]),'check'] = True
            df.loc[(df["imageName"] == dorsal_img[0]),'check'] = True
            

            result_dict["images"].append([palmar_img, dorsal_img])

            '''
                Exclude people who no longer have palm and back images to extract
            '''
            if df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('palmar')), 'check'].all() or df.loc[(df['id'] == person_id) & (df['aspectOfHand'].str.contains('dorsal')), 'check'].all():
                person_id_list = np.delete(person_id_list, np.where(person_id_list == person_id)[0])

    return result_dict

# Prepare data for subject identification
def prepare_data_SVC(csv_path:str, num_sub:int, num_img:int, isClosedSet:bool=True, num_impostors:int=0):
    df = pd.read_csv(csv_path)

    result_dict = {
        "train": None,
        "test": None
    }    
    result_dict["train"], df = prepare_data_SVC_train(df, num_sub, num_img)
    prepare_data_SVC_test(df, result_dict, num_img, isClosedSet, num_impostors)
    return result_dict

def prepare_data_SVC_train(df:pd.DataFrame, num_sub:int, num_img: int):
    result_dict = {
        "person_id": [],
        #"side": [],num_sub
        "images": []
    }    
    '''
    # Togliamo le immagini con accessori quando si fa il training in modo da evitare poi di avere un persona che non ha abbastanza immagini senza accessori
    if phase:
        palmo_df = palmo_df[palmo_df['accessories'] == 0]
        dorso_df = dorso_df[dorso_df['accessories'] == 0]
    
    palmo_df = df[df['aspectOfHand'].str.contains('palm')]
    dorso_df = df[df['aspectOfHand'].str.contains( 'dorsal')]
  
    palm_counts = palmo_df.groupby('id').size().reset_index(name='palm_count')
    dorsal_counts = dorso_df.groupby('id').size().reset_index(name='dorsal_count')
    result_df = pd.merge(palm_counts, dorsal_counts, on='id', how='outer').fillna(0)
    result_df['palm_count'] = result_df['palm_count'].astype(int)
    result_df['dorsal_count'] = result_df['dorsal_count'].astype(int)
    filtered_df = result_df[(result_df['palm_count'] >= num_img) & (result_df['dorsal_count'] >= num_img)]
    
    # Verifichiamo che ci siano almeno num_sub persone che soddisfano i criteri
    if len(filtered_df) < num_sub:
        print(f"Solo {len(filtered_df)} persone soddisfano i criteri.")
        sample_df = filtered_df  
    else:
    
        sample_df = filtered_df.sample(n=num_sub, replace=False)  

    # Id delle persone che hanno almno num_train immagini per palmo e dorso senza essere ripetute
    person_id_list = sample_df['id'].tolist()
    #print(len(person_id_list))
    '''
    # Creiamo un campo check per sapere se un'immagine è stata già presa    
    df['check'] = False
    person_id_list = np.random.choice(df['id'].unique(), size=num_sub, replace=False)

    #print(set(person_id_list))
    
    print("Prepare data train")
    # costruiamo un df con id persona e numero di immagini e tagliamo su quello
    for person_id in person_id_list:
        for _ in range (0, int((num_img/100)*70)):
            result_dict["person_id"].append(person_id)                
            palmar_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("palmar")),'imageName'].sample(n=1, replace=False).to_list()
            dorsal_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("dorsal")),'imageName'].sample(n=1, replace=False).to_list()
            '''
            df.loc[(df["imageName"] == palmar_img[0]),'check'] = True
            df.loc[(df["imageName"] == dorsal_img[0]),'check'] = True
            '''
            #result_dict["side"].append(["palmar", "dorsal"])
            result_dict["images"].append([palmar_img[0], dorsal_img[0]])

    print("Fine prepare data train")
    return result_dict, df

def prepare_data_SVC_test(df:pd.DataFrame, dict:dict, num_img: int, isClosedSet:bool=True, num_impostors:int=0):
    if isClosedSet or num_impostors == 0:
        person_id_list = set( dict["train"]["person_id"] )
    else:
        person_id_list = np.random.choice(dict["train"]["person_id"], size=len(dict["train"]["person_id"])-num_impostors, replace=False).tolist()
        impostor_list = list(set(df['id'].unique()) - set(person_id_list))
        person_id_list.extend(np.random.choice(impostor_list, size=num_impostors, replace=False).tolist())
    
    dict["test"] = {
        "person_id": [],
        "images": []
    }  

    #print(set(person_id_list))

    print("Prepare data test")
    # costruiamo un df con id persona e numero di immagini e tagliamo su quello
    for person_id in person_id_list:
        for _ in range (0, int((num_img/100)*30)):
            
            if not isClosedSet:
                # If the person is an impostor, the label is -1
                if person_id in impostor_list:
                    dict["test"]["person_id"].append(-1)
                else:
                    dict["test"]["person_id"].append(person_id)
            else:
                dict["test"]["person_id"].append(person_id)
               
            palmar_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("palmar")),'imageName'].sample(n=1, replace=False).to_list()
            dorsal_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("dorsal")),'imageName'].sample(n=1, replace=False).to_list()
            #result_dict["side"].append(["palmar", "dorsal"])
            dict["test"]["images"].append([palmar_img[0], dorsal_img[0]])
            
    print("Fine prepare data test")

import pandas as pd
import numpy as np

# Prepare data for subject identification
def prepare_data_SVC(csv_path:str, num_sub:int, num_img:int, isClosedSet:bool=True, num_impostors:int=0, perc_test:float=0.2):
    df = pd.read_csv(csv_path)

    result_dict = {
        "train": None,
        "test": None
    }    

    person_id_list = np.random.choice(df['id'].unique(), size=num_sub, replace=False).tolist()

    result_dict["train"], df = prepare_data_SVC_train(df, person_id_list, num_img, perc_test)
    result_dict["test"] = prepare_data_SVC_test(df, person_id_list, num_img, isClosedSet, num_impostors, perc_test=perc_test)
    return result_dict

def prepare_data_SVC_train(df:pd.DataFrame, person_id_list:list, num_img: int, perc_test:float=0.2):
    result_dict = {
        "person_id": [],
        "images": []
    }    
    
    print("Prepare data train")
    # costruiamo un df con id persona e numero di immagini
    for person_id in person_id_list:
        for _ in range (0, int(num_img - (num_img*perc_test))):
            result_dict["person_id"].append(person_id)                
            palmar_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("palmar")),'imageName'].sample(n=1, replace=False).to_list()
            dorsal_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("dorsal")),'imageName'].sample(n=1, replace=False).to_list()
            result_dict["images"].append([palmar_img[0], dorsal_img[0]])

    print("Fine prepare data train")
    return result_dict, df


def prepare_data_SVC_test(df:pd.DataFrame, person_id_list:list, num_img: int, isClosedSet:bool=True, num_impostors:int=0, perc_test:float=0.2):
    dict = {
        "person_id": [],
        "images": []
    }
    
    if not isClosedSet and num_impostors != 0:
        person_id_list = np.random.choice(person_id_list, size=len(person_id_list)-num_impostors, replace=False).tolist()
        impostor_list = list(set(df['id'].unique()) - set(person_id_list))
        person_id_list.extend(np.random.choice(impostor_list, size=num_impostors, replace=False).tolist())  

    print("Prepare data test")
    # costruiamo un df con id persona e numero di immagini e tagliamo su quello
    for person_id in person_id_list:
        for _ in range (0, int(num_img*perc_test)):        
            if not isClosedSet:
                # If the person is an impostor, the label is -1
                if person_id in impostor_list:
                    dict["person_id"].append(-1)
                else:
                    dict["person_id"].append(person_id)
            else:
                dict["person_id"].append(person_id)
               
            palmar_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("palmar")),'imageName'].sample(n=1, replace=False).to_list()
            dorsal_img = df.loc[(df["id"] == person_id)&(df["aspectOfHand"].str.contains("dorsal")),'imageName'].sample(n=1, replace=False).to_list()
            
            dict["images"].append([palmar_img[0], dorsal_img[0]])
            
    print("Fine prepare data test")

    return dict

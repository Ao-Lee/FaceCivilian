'''
Convert a folder of images to nd array, and save it to a file
'''

import numpy as np
from os.path import join, isfile
from os import listdir
from PIL import Image
from tqdm import tqdm
from vggface import VggFace, preprocess_input
import cfg

path_img_folder = 'E:\\DM\\Faces\\Data\\PCD\\aligned'
path_output_array = 'E:\\DM\\Faces\\Data\\PCD\\array.npy'
path_output_person_idx = 'E:\\DM\\Faces\\Data\\PCD\\person_idx.npy'

model = VggFace(cfg.dir_model_v2)

def Path2Image(path):
    im = Image.open(path)
    im = im.resize((224,224))
    im = np.array(im).astype(np.float32)
    return im
    
def SubFolder2Array(path):
    embeddings = []
    for file in listdir(path):
        current_path = join(path, file)
        if not isfile(current_path):
            continue
        img = Path2Image(current_path)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        embedding = model.predict_on_batch(img)
        embeddings.append(embedding.reshape(-1))
    
    embeddings = np.array(embeddings)
    return embeddings


def Folder2Array():
    persons = []
    embs = []
    for person_idx, subfolder in tqdm(enumerate(listdir(path_img_folder))):
        emb = SubFolder2Array(join(path_img_folder, subfolder))
        person = np.zeros(emb.shape[0], dtype=np.int)
        person[:] = person_idx
        embs.append(emb)
        persons.append(person)
        
    embs = np.concatenate(embs)
    persons = np.concatenate(persons)
    return embs, persons
    
if __name__=='__main__':
    embs, persons_idx = Folder2Array()
    np.save(path_output_array, embs)
    np.save(path_output_person_idx, persons_idx)
    
    

        



        



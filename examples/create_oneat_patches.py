import hydra
from oneat.NEATUtils.utils import save_json, load_json
from oneat.NEATUtils import MovieCreator
from oneat.NEATModels import NEATDenseVollNet
from oneat.NEATModels.TrainConfig import TrainConfig
import os
import numpy as np

def main():

    base_dir = '/path/to/base_dir'
    
    image_dir = os.path.join(base_dir, 'image_dir')
    csv_dir = os.path.join(base_dir, 'csv_dir')
    seg_dir = os.path.join(base_dir, 'seg_dir')
    save_patch_dir = os.path.join(base_dir, 'save_patch_dir')
    model_dir = '/path/to/model_dir'
    #Name of the  events
    event_type_name = ['Normal', 'Mitosis', 'Apoptosis']
    #Label corresponding to event
    event_type_label = [0, 1, 2]

    #The name appended before the CSV files
    csv_name_diff = 'ONEAT'
    size_tminus = 1
    size_tplus = 1
    tshift = 0
    imagex  = 64
    imagey = 64
    imagez = 8
    normalizeimage = True
    npz_name = 'npzfile.npz'
    npz_val_name = 'npzfile_val.npz'
    crop_size = [imagex,imagey,imagez, size_tminus,size_tplus]
    event_position_name = ["x", "y", "z", "t", "h", "w", "d", "c"]
    event_position_label = [0, 1, 2, 3, 4, 5, 6, 7]
    dynamic_config = TrainConfig(event_type_name, event_type_label, event_position_name, event_position_label)

    dynamic_json, dynamic_cord_json = dynamic_config.to_json()

    save_json(dynamic_json, model_dir + 'categories.json')

    save_json(dynamic_cord_json, model_dir + 'cord.json')
    
    MovieCreator.VolumeLabelDataSet(image_dir, 
                               seg_dir, 
                               csv_dir, 
                               save_patch_dir, 
                               event_type_name, 
                               event_type_label, 
                               csv_name_diff,
                               crop_size,
                               normalizeimage = normalizeimage,
                               tshift = tshift, 
                               dtype=np.float32)
    MovieCreator.createNPZ(save_patch_dir, axes = 'STZYXC', save_name = npz_name, save_name_val = npz_val_name)

if __name__=='__main__':
    main()  

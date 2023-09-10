import os
from oneat.NEATModels import NEATDenseVollNet
from oneat.NEATModels.config import volume_config
from oneat.NEATUtils.utils import save_json, load_json


def main():

    npz_directory = '/path/to/npzdirectory'
    model_dir = '/path/to/modeldir'
    npz_name = 'npzfile.npz'
    npz_val_name = 'npzfile_val.npz'
    #Neural network parameters
    division_categories_json = os.path.join(model_dir, 'division_categories.json')
    key_categories = load_json(division_categories_json)
    
    division_cord_json = os.path.join(model_dir, 'division_cord.json')
    key_cord = load_json(division_cord_json)

    #Number of starting convolutional filters, is doubled down with increasing depth
    startfilter = 32
    #CNN network start layer, mid layers and lstm layer kernel size
    start_kernel = 7
    mid_kernel = 3
    #Size of the gradient descent length vector, start small and use callbacks to get smaller when reaching the minima
    learning_rate = 0.001
    #For stochastic gradient decent, the batch size used for computing the gradients
    batch_size = 32
    #Training epochs, longer the better with proper chosen learning rate
    epochs = 200
    
    #The inbuilt model stride which is equal to the nulber of times image was downsampled by the network
    show = False
    stage_number = 3
    size_tminus = 1
    size_tplus = 1
    imagex = 64
    imagey = 64
    imagez = 8
    trainclass = NEATDenseVollNet
    trainconfig = volume_config
    depth = {'depth_0': 6, 'depth_1': 12, 'depth_2': 24}
    reduction = 0.5
    config= trainconfig(npz_directory = npz_directory, npz_name = npz_name, npz_val_name = npz_val_name,  
                            key_categories = key_categories, key_cord = key_cord, imagex = imagex,
                            reduction = reduction,
                            imagey = imagey, imagez = imagez, size_tminus = size_tminus, size_tplus = size_tplus, epochs = epochs,learning_rate = learning_rate,
                            depth = depth, start_kernel = start_kernel, mid_kernel = mid_kernel, stage_number = stage_number,
                            show = show,startfilter = startfilter, batch_size = batch_size)

    config_json = config.to_json()
    print(config)
    save_json(config_json, model_dir + '/' + 'parameters.json')
    Train: NEATDenseVollNet = trainclass(config, model_dir, key_categories, key_cord)
    Train.loadData()
    Train.TrainModel()

if __name__ == '__main__':
    main()  

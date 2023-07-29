import os
from oneat.NEATUtils import TrainDataMaker

def main():
    base_dir = '/path/to/base_dir'
    oneat_nuclei_train_data = os.path.join(base_dir,'timelapse_image')
    TrainDataMaker(oneat_nuclei_train_data)

if __name__=='__main__':
    main()  

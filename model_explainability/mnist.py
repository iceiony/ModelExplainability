import gzip
import numpy as np

from pathlib import Path

#read data from the files
def load_mnist_data(data_dir = './data', data_type = 'train'):

    data_dir = Path(data_dir)

    with gzip.open(data_dir / f'{data_type}-images-idx3-ubyte.gz','r') as f:
        magic_number, number_of_images, number_of_rows, number_of_columns = [int.from_bytes(f.read(4)) for i in range(4)]
        
        assert magic_number == 2051
        assert number_of_images ==  60000

        print(f'Number of Images: {number_of_images}, Image Size: {number_of_rows} x { number_of_columns }')
        
        buffer = f.read(number_of_rows * number_of_columns * number_of_images)
        images = np.frombuffer(buffer, np.ubyte).astype('int').reshape(number_of_images, number_of_rows, number_of_columns)

        del buffer


    with gzip.open(data_dir / f'{data_type}-labels-idx1-ubyte.gz','r') as f:
        magic_number, number_of_items = [int.from_bytes(f.read(4)) for i in range(2)]
        assert magic_number == 2049

        buffer = f.read(number_of_items)
        labels = np.frombuffer(buffer, np.ubyte).astype('int')

        del buffer

    return (images, labels)


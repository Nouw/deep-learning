import h5py

from src.loader import MEGDataset


def load_file(path: str):
    return h5py.File(path, 'r')

def main():
    folder = "Final Project data/Intra/train"
    dataset = MEGDataset(folder)

    print(dataset)


if __name__ == "__main__":
    main()

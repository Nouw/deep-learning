import scipy

def main():
    mat = scipy.io.loadmat('Xtrain.mat')
    print(mat)


if __name__ == "__main__":
    main()

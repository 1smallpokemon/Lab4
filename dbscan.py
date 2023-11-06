import sys

def main(fileName, threshold):
    """
    takes in filename and threshold (if no threshold was passed, threshold is None)
    """
    pass

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1], None)
    else:
        main(sys.argv[1], sys.argv[2])
import os
import argparse
import random
from tqdm import tqdm

def main():
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the --reduce_by argument
    parser.add_argument(
        '--reduce_by',
        type=float,  # or int, depending on your use case
        required=True,
        help='The value by which to reduce the charades dataset'
    )

    # Parse the arguments
    args = parser.parse_args()

    os.makedirs(f"{os.path.dirname(os.path.abspath(__file__))}/reduced/{args.reduce_by}", exist_ok=True)

    train_dir = f"{os.path.dirname(os.path.abspath(__file__))}/charades_sta_train.txt"
    test_dir = f"{os.path.dirname(os.path.abspath(__file__))}/charades_sta_test.txt"

    new_train_dir = f"{os.path.dirname(os.path.abspath(__file__))}/reduced/{args.reduce_by}/charades_sta_train.txt"
    new_test_dir = f"{os.path.dirname(os.path.abspath(__file__))}/reduced/{args.reduce_by}/charades_sta_test.txt"

    with open(train_dir, "r") as train_file:
        train_data = train_file.readlines()

        print(f"*Original training data length: {len(train_data)}")
        
        del_num = int(len(train_data) * args.reduce_by)
        for _ in range(del_num):
            del train_data[random.randint(0, len(train_data) - 1)]

        print(f"*Reduced  training data length: {len(train_data)}")
    
    with open(new_train_dir, "w") as new_train:
        for line in train_data:
            new_train.write(line)
        

    with open(test_dir, "r") as test_file:
        test_data = test_file.readlines()

        print(f"*Original test data length: {len(test_data)}")
        
        del_num = int(len(test_data) * args.reduce_by)
        for _ in range(del_num):
            del test_data[random.randint(0, len(test_data) - 1)]

        print(f"*Reduced test data length: {len(test_data)}")
    
    with open(new_test_dir, "w") as new_test:
        for line in test_data:
            new_test.write(line)

if __name__ == "__main__":
    main()

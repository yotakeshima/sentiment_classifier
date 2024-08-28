import os 

def run_training():
    os.system("python -m scripts.train_model")
def run_testing():
    os.system("python -m scripts.test_model")
def run_single_input():
    os.system("python -m scripts.single_input")
def run_cross_validate():
    os.system("python -m scripts.cross_validate")
def run_bert_model():
    os.system("python -m scripts.bert_model")

def display_menu():
    print("\nChoose and option:")
    print("1. train the model")
    print("2. Test the model")
    print("3. Test a single sentence")
    print("4. Cross Validate the model")
    print("5. Test a BERT model on the dataset")


def main():
    while True:
        display_menu()
        choice = input("Enter your choice (or 'q' to quit): ").strip()

        if choice == '1':
            run_training()
        elif choice == '2':
            run_testing()
        elif choice == '3':
            run_single_input()
        elif choice == '4':
            run_cross_validate()
        elif choice == '5':
            run_bert_model()
        elif choice.lower() == 'q':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
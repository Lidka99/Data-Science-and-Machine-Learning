from modules.agglomerative_grouping import agglomerative_grouping, visualizer
from modules.loader import read_wine, read_heart, read_customers


def choice(y):
    dataset = []
    match y:
        case '1':
            dataset = read_wine()
            print(dataset)
            agglomerative_grouping(dataset)
            visualizer(dataset, 0, 1, 2, 'Alcohol', 'Malic acid', 'Ash', 'Wine')

        case '2':
            dataset = read_heart()
            print(dataset)
            agglomerative_grouping(dataset)
            visualizer(dataset, 0, 3, 4, 'Age', 'Trestbps', 'Chol', 'Heart disease')

        case '3':
            dataset = read_customers()
            print(dataset)
            agglomerative_grouping(dataset)
            visualizer(dataset, 2, 5, 6, 'Age', 'Work Experience', 'Spending Score', 'Customer Segmenation')
        case _:
            print("ERROR")
            return



def main():

    option = input("Choose dataset\n 1. wine\n 2. heart\n 3. customers\n")
    choice(option)



# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    main()

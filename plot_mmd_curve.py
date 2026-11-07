import matplotlib.pyplot as plt

if __name__ == '__main__':
    mutag_data = [0.3087, 0.1578, 0.1251, 0.1458, 0.1414]
    bzr_data = [0.3036, 0.2741, 0.1863, 0.1724, 0.1779]
    cox2_data = [0.3108, 0.1544, 0.1432, 0.1506, 0.1466]
    dhfr_data = [0.2575, 0.2565, 0.1364, 0.1356, 0.1267]

    x = ['NeuralGW: S=0', 'NeuralGW: S=1', 'NeuralGW: S=3', 'NeuralGW: S=5', 'NeuralGW: S=9']

    plt.plot(x, mutag_data, marker='o', label='MUTAG')
    plt.plot(x, bzr_data, marker='o', label='BZR')
    plt.plot(x, cox2_data, marker='o', label='COX2')
    plt.plot(x, dhfr_data, marker='o', label='DHFR')

    plt.xlabel('Model Configuration')
    plt.ylabel('MMD Value')
    # plt.title('MMD Values Across Different Datasets and Model Configurations')
    plt.legend()
    plt.grid(True)
    plt.savefig('mmd_curve.png')
    plt.show()
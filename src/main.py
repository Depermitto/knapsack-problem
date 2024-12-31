import knapsack

FILENAMES = ['f1_l-d_kp_10_269.csv', 'f2_l-d_kp_20_878.csv', 'f3_l-d_kp_4_20.csv', 'f4_l-d_kp_4_11.csv', 'f5_l-d_kp_15_375.csv', 'f6_l-d_kp_10_60.csv', 'f7_l-d_kp_7_50.csv', 'f8_l-d_kp_23_10000.csv', 'f9_l-d_kp_5_80.csv', 'f10_l-d_kp_20_879.csv']

def main():
    for fname in FILENAMES:
        optimal, capacity, items = knapsack.read_data('data/small/' + fname)
        value, taken_items = knapsack.a_star(capacity, items)
        print(f"Expected value: {optimal}, Computed value: {value}")

if __name__ == '__main__':
    main()
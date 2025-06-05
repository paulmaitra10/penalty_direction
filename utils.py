from sklearn.model_selection import train_test_split

def split_data(X, y, train_ratio=0.85):
    return train_test_split(X, y, test_size=1 - train_ratio, random_state=42)


def get_labels_dict():
    return {
        'centre': 0,
        'left_lower': 1,
        'left_upper': 2,
        'right_lower': 3,
        'right_upper': 4
    }
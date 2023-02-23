def num_bins(data):
    return data.max() - data.min() + 1

def data_range(data) -> tuple[int, list]:
    max_ = data.max()
    min_ = data.min()

    number_of_bins = max_ - min_ + 1
    range_ = [min_, max_ + 1]
    return number_of_bins, range_
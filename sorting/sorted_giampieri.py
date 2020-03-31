
# %%
def my_sorted(unsorted_list):
    ordered_list = []
    if len(unsorted_list)==1:
        return unsorted_list.copy()
    for i in range(len(unsorted_list)):
        ordered_list.append(unsorted_list[i])
        if i==0:
            continue
        current = ordered_list[i]
        previous = ordered_list[i-1]
        if current<previous:
            'the element looking is less than the previous, invert them'
            ordered_list[i-1] = current
            ordered_list[i] = previous
            ordered_list = my_sorted(ordered_list)
    return ordered_list  


# %%
def test_actual_ordering_variant_1():
    a = [3, 2, 1]
    assert my_sorted(a) == [1, 2, 3]

def test_actual_ordering_variant_2():
    a = [5, 4, 9]
    assert my_sorted(a) == [4, 5, 9]

def test_null_list_maintained():
    a = []
    assert my_sorted(a) == []

def test_sorted_lists_stay_sorted():
    a = [1, 2, 3]
    assert my_sorted(a) == [1, 2, 3]

def test_duplicate_items_are_kept():
    a = [3, 2, 1, 2]
    assert my_sorted(a) == [1, 2, 2, 3]

def test_returns_new_list():
    a = [3, 2, 1, 2]
    b = my_sorted(a)
    assert a == [3, 2, 1, 2]
    assert a is not b

def test_bug_identity_for_len_1_lists():
    """test to avoid regression: is the list is of lenght 1
    it used to return the same list
    """
    a = [3]
    b = my_sorted(a)
    assert a == [3]
    assert a is not b
    
# %%

def  my_sorted_1(unsorted_list):
    sorted_list = []
    copia=unsorted_list.copy() #to not modify old list
    for i in unsorted_list:
        sorted_list.append(max(copia))
        copia.remove(max(copia))
    return( sorted_list )
    
def test_old_list_stays_same():
    prova = [1,100, 0, 0, 1]
    my_sorted_1(prova)
    assert prova == [1,100, 0, 0, 1]
    
def test_first_item_is_max():
    prova = my_sorted_1([1, 2, 0, 1])
    assert max(prova)==prova[0]
    
def test_last_item_is_min():
    prova = my_sorted_1([1, 2, 0, 1])
    assert min(prova)==prova[-1]
    

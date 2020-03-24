#%%file test_prova.py

def generate_state():
    return ".....0......"

def evolve(stato):
    return stato

def simulation(nsteps):
    initial_state = generate_state()
    states_seq = [initial_state]
    for i in range(nsteps):
        old_state = states_seq[-1]
        new_state = evolve(old_state)
        states_seq.append(new_state)
        print (old_state)
    return states_seq

simulation(3)
########################################################

def test_generation_valid_state():
    state = generate_state()
    assert set(state) == {'.', '0'}

a == 1
b = "a"
a=b
def test_generation_single_alive():
    state = generate_state()
    num_of_0 = sum(1 for i in state if i=='0')
    assert num_of_0 == 1

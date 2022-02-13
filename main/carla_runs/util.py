import os
import pickle


def save_tratador_and_normalizador(working_dir, name, tratador, normalizador):
    os.makedirs(os.path.join(working_dir, 'pkls'), exist_ok=True)
    with open(os.path.join(working_dir, 'pkls', f'{name}_tratador.pkl'), 'wb') as file:
        pickle.dump(tratador, file)

    with open(os.path.join(working_dir, 'pkls', f'{name}_normalizador.pkl'), 'wb') as file:
        pickle.dump(normalizador, file)

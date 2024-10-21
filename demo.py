from src.maincode.custom_functions import is_even

import streamlit as st


def app():
    st.header('Test if your number is Even 💚')
    n = st.slider('Provide an integer to test', min_value = 0, max_value = 100)
    if n:
        resp = is_even(n)
        if resp:
            st.success(f'{n} is even !')
        else:
            st.info(f'{n} is odd...')


if __name__ == '__main__':
	app()
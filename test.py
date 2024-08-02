import streamlit as st

def main():
    st.title("Debug Streamlit App")
    st.write("If you see this, Streamlit is working.")

    if st.button("Show Hello"):
        st.write("Hello World")

if __name__ == "__main__":
    main()
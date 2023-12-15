import streamlit as st  # type: ignore
import base64


def display_trending_news(
    title: str = "Trending News Time Series", gif_file: str = "viz/hi.gif"
) -> None:
    """
    Displays a demo of trending news visualization using Streamlit.

    The data is from Kaggle (https://www.kaggle.com/notlucasp/financial-news-headlines).
    The source code is available on GitHub (https://github.com/ankoroma/data-440-project).
    """
    st.title(title)
    st.markdown(
        "This is a demo of trending news visualization. The data is from [Kaggle](https://www.kaggle.com/notlucasp/financial-news-headlines)."
    )
    st.markdown(
        "The source code is available on [GitHub](https://github.com/ankoroma/data-440-project)"
    )

    file_ = open(gif_file, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="trending news gif">',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    display_trending_news()

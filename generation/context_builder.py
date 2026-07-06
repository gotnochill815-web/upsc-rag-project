"""
Build context for the language model.
"""

def build_context(df, max_chars=800):
    """
    Build context from retrieved documents.

    Parameters
    ----------
    df : pandas.DataFrame
        Retrieved documents.

    max_chars : int
        Maximum number of characters to keep from each document.

    Returns
    -------
    str
        Formatted context.
    """

    context = ""

    for i, row in enumerate(df.itertuples(index=False), start=1):

        text = str(row.text)[:max_chars]

        context += f"""
Document {i}

Title:
{row.title}

Source:
{row.source}

Content:
{text}

==================================================

"""

    return context
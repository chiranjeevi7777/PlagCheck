import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_pdf(filename="test_doc.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        spaceAfter=20
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        spaceAfter=12
    )
    
    story = []
    
    story.append(Paragraph("A Study on Deep Learning and Artificial Intelligence", title_style))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(
        "Deep learning is a subset of machine learning, which is in turn a subset of artificial intelligence. "
        "Deep learning is based on representation learning using artificial neural networks. "
        "Unlike traditional machine learning algorithms, deep learning algorithms can automatically learn features from raw data, "
        "making them highly effective for computer vision, natural language processing, and speech recognition tasks.",
        body_style
    ))
    
    story.append(Paragraph(
        "Artificial neural networks are inspired by the biological neural networks in human brains. "
        "A network consists of layers of interconnected nodes, called artificial neurons. "
        "Information flows through the network from the input layer, through one or more hidden layers, to the output layer. "
        "Each connection has an associated weight that is adjusted during the training process to minimize prediction errors.",
        body_style
    ))
    
    story.append(Paragraph(
        "Plagiarism detection systems compare a given document against a corpus of existing texts to identify similarities and copying. "
        "With the advent of advanced large language models, detecting AI-generated text has become an essential counterpart to traditional plagiarism checks. "
        "Modern systems utilize hybrid approaches combining lexical matching with style analysis to produce comprehensive integrity reports.",
        body_style
    ))
    
    doc.build(story)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    create_pdf()

# Imports the Google Cloud client library
from google.cloud import vision


# client = vision.ImageAnnotatorClient()
# file_url = "data/IAM_lines/a01/a01-000u/a01-000u-00.png"
# image = vision.Image()
# image.source.image_uri = file_uri


def run_quickstart() -> vision.EntityAnnotation:
    """Provides a quick start example for Cloud Vision."""

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # The URI of the image file to annotate
    # file_uri = "gs://cloud-samples-data/vision/label/wakeupcat.jpg"
    file_uri = "data/IAM_lines/a01/a01-000u/a01-000u-00.png"

    image = vision.Image()
    image.source.image_uri = file_uri

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    print("Labels:")
    for label in labels:
        print(label.description)

    return labels


# run_quickstart()
#################################################


def detect_document(path):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print(f"\nBlock confidence: {block.confidence}\n")

            for paragraph in block.paragraphs:
                print("Paragraph confidence: {}".format(paragraph.confidence))

                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    print(
                        "Word text: {} (confidence: {})".format(
                            word_text, word.confidence
                        )
                    )

                    for symbol in word.symbols:
                        print(
                            "\tSymbol: {} (confidence: {})".format(
                                symbol.text, symbol.confidence
                            )
                        )

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )


# detect_document("data/IAM_lines/a01/a01-000u/a01-000u-00.png")
detect_document("attachments/eddy-pencil-handwriting.jpeg")

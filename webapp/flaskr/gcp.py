from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename  
from flaskr.auth import login_required
from flaskr.db import get_db
from google.cloud import vision
import os


def detect_document(path):
    """Detects document features in an image."""
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ".creds/farmdocs-7e1092c19709.json"
    client = vision.ImageAnnotatorClient()
    with open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    words = "" # in string format
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
                    words += word_text + " "
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
    return words


bp = Blueprint('gcp', __name__)
UPLOAD_FOLDER = 'flaskr/static/uploads/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @bp.route('/', methods=['GET', 'POST'])
@bp.route('/' , methods=['GET', 'POST'])
def index():
    search = request.form.get('search', '')
    db = get_db()
    if not search:
        posts = db.execute(
            'SELECT p.id, title, img_path, gcp_output, created, author_id, username'
            ' FROM post p JOIN user u ON p.author_id = u.id'
            ' ORDER BY created DESC'
        ).fetchall()

    else:
        query = """
            SELECT p.id, title, img_path, gcp_output, created, author_id, username
            FROM post p 
            JOIN user u ON p.author_id = u.id
            WHERE title LIKE :search OR gcp_output LIKE :search
            OR img_path LIKE :file_search
            ORDER BY created DESC
        """
        # Handle file search for various extensions
        if search.startswith("*."):
            file_extension = search[2:]  # Extract file extension (e.g., "jpg", "png")
            file_search = f'%.{file_extension}'
        else:
            file_search = f'%{search}%'
        posts = db.execute(query, {'search': f'%{search}%', 'file_search': file_search}).fetchall()  
        
    return render_template('gcp/index.html', posts=posts)

@bp.route('/create', methods=('GET', 'POST'))
@login_required
def create():
    if request.method == 'POST':
        title = request.form['title']
        file = request.files['image']
        error = None

        if not title:
            error = 'Title is required.'
        elif file.filename == '':
            error = 'No selected file.'
        elif not allowed_file(file.filename):
            error = 'Invalid file type.'

        if error is not None:
            flash(error)

        else:
            filename = secure_filename(file.filename)
            img_path = filename
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            root_img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(root_img_path)  # Save the file in the flaskr/static/uploads/images directory
            gcp_output = detect_document(root_img_path)

            db = get_db()
            db.execute( 
                'INSERT INTO post (title, img_path, gcp_output, author_id)'
                ' VALUES (?, ?, ?, ?)',
                (title, img_path, gcp_output, g.user['id'])
            )
            db.commit()
            return redirect(url_for('gcp.index'))

    return render_template('gcp/create.html')

def get_post(id, check_author=True):
    post = get_db().execute(
        'SELECT p.id, title, img_path, gcp_output, created, author_id, username'
        ' FROM post p JOIN user u ON p.author_id = u.id'
        ' WHERE p.id = ?',
        (id,)
    ).fetchone()

    if post is None:
        abort(404, f"Post id {id} doesn't exist.")

    if check_author and post['author_id'] != g.user['id']:
        abort(403)

    return post

@bp.route('/<int:id>/update', methods=('GET', 'POST'))
@login_required
def update(id):
    post = get_post(id)

    if request.method == 'POST':
        title = request.form['title']
        gcp_output = request.form['gcp_output']
        error = None

        if not title:
            error = 'Title is required.'
        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                'UPDATE post SET title = ?, gcp_output = ?'
                ' WHERE id = ?',
                (title, gcp_output, id)
            )
            db.commit()
            return redirect(url_for('gcp.index'))

    return render_template('gcp/update.html', post=post)

@bp.route('/<int:id>/delete', methods=('POST',))
@login_required
def delete(id):
    get_post(id)
    db = get_db()
    db.execute('DELETE FROM post WHERE id = ?', (id,))
    db.commit()
    return redirect(url_for('gcp.index'))
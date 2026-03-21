from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
from chessboard_split import get_tiles
import chess
import chess.engine
from model import CNN
import torch
import torchvision
from huggingface_hub import hf_hub_download

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    print("=== Form Submission Received ===")
    all_tiles = []

    # Image upload
    if 'boardImage' in request.files:
        image = request.files['boardImage']
        print(f"Image uploaded: {image.filename}")
        all_tiles = get_tiles(image, False)

    else:
        print("No image uploaded")

    # Turn to move
    turn = request.form.get('turn', 'not specified')
    print(f"Turn to move: {turn}")

    # Castling rights
    white_kingside = request.form.get('whiteKingside') == 'on'
    white_queenside = request.form.get('whiteQueenside') == 'on'
    black_kingside = request.form.get('blackKingside') == 'on'
    black_queenside = request.form.get('blackQueenside') == 'on'
    print(f"Castling Rights - White: Kingside={white_kingside}, Queenside={white_queenside}")
    print(f"Castling Rights - Black: Kingside={black_kingside}, Queenside={black_queenside}")

    # Move counts
    move_count = request.form.get('moveCount', 'not specified')
    half_move_count = request.form.get('halfMoveCount', 'not specified')
    print(f"Move count: {move_count}")
    print(f"Half-move count: {half_move_count}")

    # En passant
    en_passant = request.form.get('enPassant', 'not specified')
    print(f"En passant target: {en_passant}")

    print("=== End of Form Submission ===\n")

    label_to_piece = {
        0: "b",
        1: "k",
        2: "n",
        3: "p",
        4: "q",
        5: "r",
        7: "B",
        8: "K",
        9: "N",
        10: "P",
        11: "Q",
        12: "R"
    }

    model = CNN()
    model_path = hf_hub_download(repo_id="advaitbhowmik/Chess_Piece_Detection", filename="model_weights.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Grayscale(num_output_channels=1),  # Match in_channels=1
        torchvision.transforms.Resize((224, 224)),  # Match training resolution
        torchvision.transforms.ToTensor(),  # Converts to [0, 1] range
    ])

    row = ""
    column = 1
    fen = ""
    spaces = 0

    for tile in all_tiles:
        piece = transform(tile)
        piece_tensor = piece.unsqueeze(0)
        piece_label = torch.argmax(model(piece_tensor), 1)
        row += str(piece_label.item()) + " "
        if piece_label.item() == 6:
            spaces += 1
        else:
            if spaces > 0:
                fen += str(spaces)
                spaces = 0
            fen += label_to_piece[piece_label.item()]
        column += 1
        if column == 9:
            column = 1
            print(row)
            row = ""
            if spaces > 0:
                fen += str(spaces)
                spaces = 0
            fen += "/"

    fen = fen[:-1]
    if turn == 'white':
        fen += " w "
    else:
        fen += " b "

    castling_rights = ""
    if white_kingside:
        castling_rights += "K"
    if white_queenside:
        castling_rights += "Q"
    if black_kingside:
        castling_rights += "k"
    if black_queenside:
        castling_rights += "q"

    if castling_rights == "":
        fen += "- "
    else:
        fen += f"{castling_rights} "


    if en_passant:
        fen += en_passant
    else:
        fen += "- "

    fen += f"{move_count} {half_move_count}"

    engine = chess.engine.SimpleEngine.popen_uci("./stockfish/stockfish-ubuntu-x86-64-avx2")
    board = chess.Board(fen)

    info = engine.analyse(board, chess.engine.Limit(time=0.1))

    best_move = info["pv"][0]
    print(f"Play {board.san(best_move)}")

    engine.quit()

    print("ALL PROCESSES COMPLETED")

    return jsonify({"best_move": f"{board.san(best_move)}"})

app.run(host = '0.0.0.0')

import sys
from server import app, initialize_models_and_data

if __name__ == '__main__':
    print("=" * 75)
    print("PRODUCTION FLASK WEB APPLICATION - MINE VEHICLE FOG SAFETY")
    print("=" * 75)
    initialize_models_and_data()
    print("\nStarting Flask server on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=False)

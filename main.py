from data_loader import load_data
from model_builder import build_model
from train import train_model, evaluate_model, plot_results
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Set file paths for local environment
    train_path = 'Training'
    test_path = 'Testing'

    # Load data
    train_images, train_labels, test_images, test_labels = load_data(train_path, test_path)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=44)

    # Build model
    model = build_model()

    # Train the model
    history, history_fine = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Plot results
    plot_results(history)

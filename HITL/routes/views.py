import os
import threading
from flask import render_template, request, jsonify, send_from_directory
from routes import views_bp
from configurations import config


# This will be set by app.py
sl_detector = None
_submit_lock = threading.Lock()


def init_views(detector):
    """Initialize view routes with detector instance"""
    global sl_detector
    sl_detector = detector


def _directory_with_visualization_file(filename):
    """
    Absolute path to a results round_* directory that contains filename.
    Prefer sl_detector.round_save_path when it contains the file; otherwise
    use the highest round_N that has the file (handles stale path / restarts).
    """
    results_path = getattr(sl_detector, 'results_path', None)
    if not results_path:
        results_path = os.path.abspath(os.path.expanduser(config.results_path))
    else:
        results_path = os.path.abspath(results_path)
    if not os.path.isdir(results_path):
        return None

    rp = getattr(sl_detector, 'round_save_path', None)
    if rp:
        d = os.path.abspath(rp)
        if os.path.isfile(os.path.join(d, filename)):
            return d

    best_dir = None
    best_num = -1
    for dirname in os.listdir(results_path):
        if not dirname.startswith('round_'):
            continue
        try:
            n = int(dirname.split('_')[1])
        except (IndexError, ValueError):
            continue
        d = os.path.join(results_path, dirname)
        if os.path.isfile(os.path.join(d, filename)) and n > best_num:
            best_num = n
            best_dir = d
    if best_dir is not None:
        return best_dir

    root_file = os.path.join(results_path, filename)
    if os.path.isfile(root_file):
        return results_path
    return None


@views_bp.route('/')
def index():
    return render_template('index.html')


@views_bp.route('/gallery')
def gallery():
    return render_template('gallery.html')


@views_bp.route('/app/submit_selections', methods=['POST'])
def submit_selections():
    if not _submit_lock.acquire(blocking=False):
        return jsonify({'success': False, 'error': 'Submit already in progress'}), 429

    print('\n=== /app/submit_selections called ===')

    try:
        data = request.json
        sl_names = data.get('sl_names', [])
        non_sl_names = data.get('non_sl_names', [])

        print(f'Received SL names: {sl_names}')
        print(f'Received Non-SL names: {non_sl_names}')

        sl_detector.add_selections(sl_names, non_sl_names)

        print(f'After add_selections - SL count: {len(sl_detector.selected_sl_names)}, Non-SL count: {len(sl_detector.selected_non_sl_names)}')
        num_submission_train = sl_detector.num_submission_train

        response_data = {
            'success': True,
            'round': sl_detector.current_round,
            'sl_count': len(sl_detector.selected_sl_names),
            'non_sl_count': len(sl_detector.selected_non_sl_names),
            'testing_sl_count': len(sl_detector.testing_sl_names),
            'testing_non_sl_count': len(sl_detector.testing_non_sl_names),
            'recovered_count': len(sl_detector.injected_recovery),
            'total_submissions': sl_detector.total_submissions,
            'available_count': sl_detector.get_available_galaxies(),
        }

        if sl_detector.total_submissions % num_submission_train == 0 and sl_detector.total_submissions > 0:
            response_data['should_train'] = True
        else:
            response_data['should_train'] = False
            if sl_detector.model_trained:
                names, scores = sl_detector.get_images()
            else:
                names, scores = sl_detector.get_random_batch(10)
            response_data['galaxy_names'] = names
            response_data['scores'] = scores
            response_data['model_trained'] = sl_detector.model_trained

        return jsonify(response_data)
    except Exception as e:
        print(f'Error submitting selections: {str(e)}')
        return jsonify({'error': str(e)}), 500
    finally:
        _submit_lock.release()


@views_bp.route('/images/<filename>')
def serve_image(filename):
    # Add .jpg extension if not present
    if not filename.endswith('.jpg'):
        filename = f"{filename}.jpg"
    return send_from_directory(sl_detector.images_path, filename)

@views_bp.route('/app/reset_model', methods=['POST'])
def reset_model():
    print('\n=== /app/reset_model called ===')
    try:
        sl_detector.reset_model()
        return jsonify({'success': True})
    except Exception as e:
        print(f'Error resetting model: {str(e)}')
        return jsonify({'error': str(e)}), 500

@views_bp.route('/app/reset_round', methods=['POST'])
def reset_round():
    print('\n=== /app/reset_round called ===')
    
    try:
        sl_detector.reset_round()
        
        print(f'After reset_round - SL count: {len(sl_detector.selected_sl_names)}, Non-SL count: {len(sl_detector.selected_non_sl_names)}')
        
        # Load next batch of images
        if sl_detector.model_trained:
            names, scores = sl_detector.get_images()
        else:
            names, scores = sl_detector.get_random_batch(10)
        
        response_data = {
            'success': True,
            'round': sl_detector.current_round,
            'sl_count': len(sl_detector.selected_sl_names),
            'non_sl_count': len(sl_detector.selected_non_sl_names),
            'testing_sl_count': len(sl_detector.testing_sl_names),
            'testing_non_sl_count': len(sl_detector.testing_non_sl_names),
            'recovered_count': len(sl_detector.injected_recovery),
            'total_submissions': sl_detector.total_submissions,
            'available_count': sl_detector.get_available_galaxies(),
            'galaxy_names': names,
            'scores': scores,
            'model_trained': sl_detector.model_trained,
        }
        
        return jsonify(response_data)
    except Exception as e:
        print(f'Error resetting round: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@views_bp.route('/visualizations/<filename>')
def serve_visualization(filename):
    base = os.path.basename(filename)
    if base != filename or '..' in filename or filename.startswith('.'):
        return "Invalid filename", 400
    try:
        directory = _directory_with_visualization_file(filename)
        if not directory:
            print(f'serve_visualization: no directory with {filename!r} under {config.results_path!r}')
            return "No visualizations available", 404
        return send_from_directory(directory, filename)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error serving visualization: {str(e)}", 500

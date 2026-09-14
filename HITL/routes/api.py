import os
import json
import threading
import numpy as np
from flask import request, jsonify, send_from_directory
from routes import api_bp
from configurations import config


# This will be set by app.py
sl_detector = None
_training_lock = threading.Lock()


def init_api(detector):
    """Initialize API routes with detector instance"""
    global sl_detector
    sl_detector = detector


@api_bp.route('/get_random_batch')
def api_get_random_batch():
    names, scores = sl_detector.get_random_batch(10)
    
    try:
        return jsonify({
            'success': True,
            'galaxy_names': names,
            'scores': scores, 
            'round': sl_detector.current_round,
            'sl_count': len(sl_detector.selected_sl_names),
            'non_sl_count': len(sl_detector.selected_non_sl_names),
            'testing_sl_count': len(sl_detector.testing_sl_names),
            'testing_non_sl_count': len(sl_detector.testing_non_sl_names),
            'recovered_count': len(sl_detector.injected_recovery),
            'total_submissions': sl_detector.total_submissions,
            'available_count': sl_detector.get_available_galaxies(),
            'model_trained': sl_detector.model_trained,
        })
    except Exception as e:
        print('Error geting random batch.')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/get_images')
def api_get_images():
    try:
        print(f'\n=== /api/get_images called ===')
        print(f'Model trained: {sl_detector.model_trained}')
        
        if sl_detector.model_trained:
            print('Using get_images() (smart selection)')
            names, scores = sl_detector.get_images()
        else:
            print('Using get_random_batch() (random selection)')
            names, scores = sl_detector.get_random_batch(10)
        
        print(f'Returning {len(names)} images')
        
        return jsonify({
            'success': True,
            'galaxy_names': names,
            'scores': scores,
            'round': sl_detector.current_round,
            'sl_count': len(sl_detector.selected_sl_names),
            'non_sl_count': len(sl_detector.selected_non_sl_names),
            'testing_sl_count': len(sl_detector.testing_sl_names),
            'testing_non_sl_count': len(sl_detector.testing_non_sl_names),
            'recovered_count': len(sl_detector.injected_recovery),
            'total_submissions': sl_detector.total_submissions,
            'available_count': sl_detector.get_available_galaxies(),
            'model_trained': sl_detector.model_trained,
        })
    except Exception as e:
        print('Error getting images:', str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@api_bp.route('/run_training', methods=['POST'])
def run_training():
    if not _training_lock.acquire(blocking=False):
        return jsonify({'success': False, 'error': 'Training already in progress'}), 429

    try:
        result = sl_detector.train_embeddings()
        return jsonify(result)
    except Exception as e:
        print('Error running training:', str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        _training_lock.release()


@api_bp.route('/get_status')
def get_status():
    # Read metrics directly from detector attributes.
    ranks_injections = getattr(sl_detector, 'ranks_injections', {}) or {}
    injected_recovery = getattr(sl_detector, 'injected_recovery', {}) or {}
    separation_accuracy = getattr(sl_detector, 'separation_accuracy', None)
    purity = getattr(sl_detector, 'purity', None)

    # Keep runtime purity updated if possible.
    if purity is None and sl_detector.stats.get('num_over_min_score_cowls', 0) > 0:
        purity = len(sl_detector.selected_sl_names) / sl_detector.stats['num_over_min_score_cowls']
        sl_detector.purity = purity

    # Fallback to latest records.json metrics when runtime state is not populated
    # (e.g., app restart without checkpoint load).
    if separation_accuracy is None or ranks_injections.get('median_rank') is None or purity is None:
        latest_records = None
        latest_round_num = -1
        if os.path.exists(config.results_path):
            for dirname in os.listdir(config.results_path):
                if dirname.startswith('round_'):
                    try:
                        round_num = int(dirname.split('_')[1])
                    except Exception:
                        continue
                    records_path = os.path.join(config.results_path, dirname, 'records.json')
                    if os.path.exists(records_path) and round_num > latest_round_num:
                        latest_round_num = round_num
                        latest_records = records_path

        if latest_records is not None:
            try:
                with open(latest_records, 'r') as f:
                    rec = json.load(f)
                if separation_accuracy is None:
                    separation_accuracy = rec.get('separation_accuracy')
                    sl_detector.separation_accuracy = separation_accuracy
                if ranks_injections.get('median_rank') is None:
                    ranks_injections = rec.get('ranks_injections', ranks_injections) or ranks_injections
                    sl_detector.ranks_injections = ranks_injections
                if purity is None:
                    purity = rec.get('purity')
                    sl_detector.purity = purity
                if len(injected_recovery) == 0:
                    injected_recovery = rec.get('injected_recovery', {}) or {}
                    sl_detector.injected_recovery = injected_recovery
            except Exception:
                pass

    return jsonify({
        'success': True,
        'round': sl_detector.current_round,
        'sl_count': len(sl_detector.selected_sl_names),
        'non_sl_count': len(sl_detector.selected_non_sl_names),
        'testing_sl_count': len(sl_detector.testing_sl_names),
        'testing_non_sl_count': len(sl_detector.testing_non_sl_names),
        'recovered_count': len(injected_recovery),
        'total_submissions': sl_detector.total_submissions,
        'available_count': sl_detector.get_available_galaxies(),
        'model_trained': sl_detector.model_trained,
        'separation_accuracy': separation_accuracy,
        'median_injection_rank': ranks_injections.get('median_rank'),
        'injected_recovered_count': len(injected_recovery),
        'purity': purity,
        'scoring_in_progress': getattr(sl_detector, 'scoring_in_progress', False),
        'scoring_last_error': getattr(sl_detector, 'scoring_last_error', None),
        'scoring_job_seq': getattr(sl_detector, 'scoring_job_seq', 0),
        'scoring_completed_seq': getattr(sl_detector, 'scoring_completed_seq', 0),
    })


@api_bp.route('/get_gallery_data')
def get_gallery_data():
    try:
        print('\n=== /api/get_gallery_data called ===')
        
        # Separate confirmed and user-selected
        confirmed_sl_names = set(sl_detector.cowls_sl_names)
        testing_sl_names = list(getattr(sl_detector, 'testing_sl_names', []) or [])
        recovered_records = list(getattr(sl_detector, 'injected_recovery', {}) or [])
        recovered_testing_sl_names = {
            name for name in sl_detector.injected_recovery.keys()
        }
        
        # Preserve selection order for user-selected items
        user_selected_sl_names = [name for name in sl_detector.selected_sl_names if name not in confirmed_sl_names]
        user_selected_non_sl_names = sl_detector.selected_non_sl_names
        
        print(f'Confirmed SL: {len(confirmed_sl_names)}')
        print(f'User-selected SL: {len(user_selected_sl_names)}')
        print(f'User-selected non-SL: {len(user_selected_non_sl_names)}')
        if len(user_selected_non_sl_names) > 0:
            print(f'First 3 non-SL names: {user_selected_non_sl_names[:3]}')
        
        # Define grade order for sorting confirmed items (high to low)
        high_grades = ['M25'] + [f'S{i:02d}' for i in range(12, 0, -1)]
        grade_order = {grade: i for i, grade in enumerate(high_grades)}
        
        # Create confirmed SL items sorted by grade
        confirmed_items = []
        for name in sl_detector.cowls_sl_names:
            confirmed_items.append({
                'name': name,
                'type': 'sl',
                'is_confirmed': True,
                'grade': sl_detector.name_to_score.get(name, 'N/A')
            })
        
        # Sort confirmed items by grade (high to low)
        confirmed_items.sort(key=lambda x: (grade_order.get(x['grade'], 999), x['name']))
        
        # Create user-selected SL items in selection order
        user_selected_sl_items = []
        for name in user_selected_sl_names:
            user_selected_sl_items.append({
                'name': name,
                'type': 'sl',
                'is_confirmed': False,
                'grade': sl_detector.name_to_score.get(name, 'N/A')
            })
        
        # Create user-selected non-SL items in selection order
        user_selected_non_sl_items = []
        for name in user_selected_non_sl_names:
            user_selected_non_sl_items.append({
                'name': name,
                'type': 'non_sl',
                'is_confirmed': False,
                'grade': 'N/A'
            })

        testing_sl_items = []
        for name in testing_sl_names:
            testing_sl_items.append({
                'name': name,
                'type': 'testing_sl',
                'is_confirmed': False,
                'is_testing_sl': True,
                'is_recovered': name in recovered_testing_sl_names,
                'grade': sl_detector.name_to_score.get(name, 'N/A')
            })

        testing_non_sl_names = list(getattr(sl_detector, 'testing_non_sl_names', []) or [])
        testing_non_sl_items = []
        for name in testing_non_sl_names:
            testing_non_sl_items.append({
                'name': name,
                'type': 'testing_non_sl',
                'is_confirmed': False,
                'is_testing_non_sl': True,
                'grade': 'N/A'
            })
        
        # Combine: confirmed SL first, then user-selected SL, then user-selected non-SL
        gallery_items = confirmed_items + user_selected_sl_items + user_selected_non_sl_items
        
        print(f'Total gallery items: {len(gallery_items)}')
        print(f'  - Confirmed SL items: {len(confirmed_items)}')
        print(f'  - User-selected SL items: {len(user_selected_sl_items)}')
        print(f'  - User-selected non-SL items: {len(user_selected_non_sl_items)}')
        
        return jsonify({
            'success': True,
            'items': gallery_items.tolist() if isinstance(gallery_items, np.ndarray) else gallery_items,
            'testing_sl_items': testing_sl_items,
            'testing_non_sl_items': testing_non_sl_items,
            'total_count': len(gallery_items),
            'confirmed_count': len(confirmed_sl_names),
            'selected_sl_count': len(user_selected_sl_names),
            'selected_non_sl_count': len(user_selected_non_sl_names),
            'testing_sl_count': len(testing_sl_items),
            'testing_sl_recovered_count': int(sum(1 for item in testing_sl_items if item['is_recovered'])),
            'testing_non_sl_count': len(testing_non_sl_items),
        })
    except Exception as e:
        print('Error getting gallery data:', str(e))
        return jsonify({'error': str(e)}), 500

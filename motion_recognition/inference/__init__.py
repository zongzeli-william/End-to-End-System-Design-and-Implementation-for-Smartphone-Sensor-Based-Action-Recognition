def get_realtime_predictor():
    from .realtime_predictor import RealtimePredictor
    return RealtimePredictor

def predict(sensor_data):
    from .realtime_predictor import predict as p
    return p(sensor_data)

def test_connection():
    from .realtime_predictor import test_connection as tc
    return tc()

def clear_system():
    from .realtime_predictor import reset as rs
    return rs()

__all__ = ['predict', 'test_connection', 'clear_system', 'get_realtime_predictor']
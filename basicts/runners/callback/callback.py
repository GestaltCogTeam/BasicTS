class BasicTSCallback:
    """
    Base class for all BasicTS callbacks.
    """

    def on_train_start(self, runner, *args, **kwargs):
        pass

    def on_train_end(self, runner, *args, **kwargs):
        pass

    def on_epoch_start(self, runner, *args, **kwargs):
        pass

    def on_epoch_end(self, runner, *args, **kwargs):
        pass

    def on_step_start(self, runner, *args, **kwargs):
        pass

    def on_step_end(self, runner, *args, **kwargs):
        pass

    def on_validate_start(self, runner, *args, **kwargs):
        pass

    def on_validate_end(self, runner, *args, **kwargs):
        pass

    def on_test_start(self, runner, *args, **kwargs):
        pass

    def on_test_end(self, runner, *args, **kwargs):
        pass

    def on_compute_loss(self, runner, *args, **kwargs):
        pass

    def on_backward(self, runner, *args, **kwargs):
        pass

    def on_optimizer_step(self, runner, *args, **kwargs):
        pass


class BasicTSCallbackHandler:
    """
    Handler for BasicTS callbacks.
    """

    def __init__(self, callbacks: list[BasicTSCallback] = None):
        self.callbacks = callbacks if callbacks is not None else []

    def trigger(self, event_name: str, runner, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, event_name, None)
            if method is not None:
                method(runner, *args, **kwargs)

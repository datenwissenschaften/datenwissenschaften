from types import SimpleNamespace

from datenwissenschaften.persistence import RedisStore


def test_delete_prefix_removes_namespace_and_nested_keys():
    deleted = []
    redis = SimpleNamespace(
        scan_iter=lambda *, match: iter(
            [
                b"datenwissenschaften:state:Game:Level:door",
                b"datenwissenschaften:state:Game:Level:score",
            ]
        ),
        delete=lambda *keys: deleted.append(keys),
    )
    store = RedisStore.__new__(RedisStore)
    store._redis = redis
    store._prefix = "datenwissenschaften"

    store.delete_prefix("state", "Game")

    assert deleted == [
        (
            b"datenwissenschaften:state:Game:Level:door",
            b"datenwissenschaften:state:Game:Level:score",
        ),
        ("datenwissenschaften:state:Game",),
    ]

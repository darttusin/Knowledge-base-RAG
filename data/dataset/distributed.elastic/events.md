Events 
=================================================================================

Module contains events processing mechanisms that are integrated with the standard python logging. 

Example of usage: 

```
from torch.distributed.elastic import events

event = events.Event(
    name="test_event", source=events.EventSource.WORKER, metadata={...}
)
events.get_logging_handler(destination="console").info(event)

```

API Methods 
----------------------------------------------------------

torch.distributed.elastic.events. record ( *event*  , *destination = 'null'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/elastic/events/__init__.py#L75) 
:

torch.distributed.elastic.events. construct_and_record_rdzv_event ( *run_id*  , *message*  , *node_state*  , *name = ''*  , *hostname = ''*  , *pid = None*  , *master_endpoint = ''*  , *local_id = None*  , *rank = None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/elastic/events/__init__.py#L83) 
:   Initialize rendezvous event object and record its operations. 

Parameters
:   * **run_id** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – The run id of the rendezvous.
* **message** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – The message describing the event.
* **node_state** ( *NodeState*  ) – The state of the node (INIT, RUNNING, SUCCEEDED, FAILED).
* **name** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – Event name. (E.g. Current action being performed).
* **hostname** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – Hostname of the node.
* **pid** ( *Optional* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]*  ) – The process id of the node.
* **master_endpoint** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – The master endpoint for the rendezvous store, if known.
* **local_id** ( *Optional* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]*  ) – The local_id of the node, if defined in dynamic_rendezvous.py
* **rank** ( *Optional* *[* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *]*  ) – The rank of the node, if known.

Returns
:   None

Return type
:   None

Example 

```
>>> # See DynamicRendezvousHandler class
>>> def _record(
...     self,
...     message: str,
...     node_state: NodeState = NodeState.RUNNING,
...     rank: Optional[int] = None,
... ) -> None:
...     construct_and_record_rdzv_event(
...         name=f"{self.__class__.__name__}.{get_method_name()}",
...         run_id=self._settings.run_id,
...         message=message,
...         node_state=node_state,
...         hostname=self._this_node.addr,
...         pid=self._this_node.pid,
...         local_id=self._this_node.local_id,
...         rank=rank,
...     )

```

torch.distributed.elastic.events. get_logging_handler ( *destination = 'null'* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/elastic/events/handlers.py#L19) 
:   Return type
:   [*Handler*](https://docs.python.org/3/library/logging.html#logging.Handler "(in Python v3.13)")

Event Objects 
--------------------------------------------------------------

*class* torch.distributed.elastic.events.api. Event ( *name*  , *source*  , *timestamp=0*  , *metadata=<factory>* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/elastic/events/api.py#L28) 
:   The class represents the generic event that occurs during the torchelastic job execution. 

The event can be any kind of meaningful action. 

Parameters
:   * **name** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  ) – event name.
* **source** ( [*EventSource*](#torch.distributed.elastic.events.api.EventSource "torch.distributed.elastic.events.api.EventSource")  ) – the event producer, e.g. agent or worker
* **timestamp** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  ) – timestamp in milliseconds when event occurred.
* **metadata** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.13)") *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* [*Union*](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)") *[* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)") *,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)") *,* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)") *,* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)") *,* *NoneType* *]* *]*  ) – additional data that is associated with the event.

*class* torch.distributed.elastic.events.api. EventSource ( *value*  , *names=<not given>*  , **values*  , *module=None*  , *qualname=None*  , *type=None*  , *start=1*  , *boundary=None* ) [source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/distributed/elastic/events/api.py#L21) 
:   Known identifiers of the event producers.

torch.distributed.elastic.events.api. EventMetadataValue 
:   alias of [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional "(in Python v3.13)")  [ [`Union`](https://docs.python.org/3/library/typing.html#typing.Union "(in Python v3.13)")  [ [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.13)")  , [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.13)")  , [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.13)")  , [`bool`](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")  ]]


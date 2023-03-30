import yamlu

POOL_CATEGORIES = [
    "Pool",
    "VerticalPool",
    "CollapsedPool",
    "CollapsedVerticalPool"
]
LANE_CATEGORIES = [
    "Lane",
    "VerticalLane"
]
PARTICIPANT_CATEGORIES = [
    "processparticipant",
    "ChoreographyParticipant",
    "Participant"
]
COLLABORATION_CATEGORIES = POOL_CATEGORIES + LANE_CATEGORIES + PARTICIPANT_CATEGORIES

TASK = "Task"
CHOREOGRAPHY_TASK = "ChoreographyTask"
SUBPROCESS_CATEGORIES = [
    "CollapsedSubprocess",
    "Subprocess",
    "EventSubprocess",
    "CollapsedEventSubprocess"
]
ACTIVITY_CATEGORIES = [TASK] + SUBPROCESS_CATEGORIES + [CHOREOGRAPHY_TASK]

MESSAGE_EVENTS = [
    "StartMessageEvent",
    "IntermediateMessageEventCatching",
    "IntermediateMessageEventThrowing",
    "EndMessageEvent",
]
TIMER_START_EVENT = "StartTimerEvent"
TIMER_INTERMEDIATE_EVENT = "IntermediateTimerEvent"
TIMER_EVENTS = [TIMER_START_EVENT, TIMER_INTERMEDIATE_EVENT]
ERROR_EVENTS = [
    "StartErrorEvent",
    "IntermediateErrorEvent",
    "EndErrorEvent"
]
ESCALATION_EVENTS = [
    "StartEscalationEvent",
    "IntermediateEscalationEvent",
    "IntermediateEscalationEventThrowing",
    "EndEscalationEvent"
]
SIGNAL_EVENTS = [
    "StartSignalEvent",
    "IntermediateSignalEventCatching",
    "IntermediateSignalEventThrowing",
    "EndSignalEvent"
]
MULTIPLE_EVENTS = [
    "StartMultipleEvent",
    "IntermediateMultipleEventCatching",
    "IntermediateMultipleEventThrowing",
    "StartParallelMultipleEvent",
    "IntermediateParallelMultipleEventCatching",
    "EndMultipleEvent"
]
CONDITIONAL_EVENTS = [
    "StartConditionalEvent",
    "IntermediateConditionalEvent"
]
LINK_EVENTS = [
    "IntermediateLinkEventThrowing",
    "IntermediateLinkEventCatching"
]
CANCEL_EVENTS = [
    "IntermediateCancelEvent",
    "EndCancelEvent"
]
COMPENSATION_EVENTS = [
    "StartCompensationEvent",
    "IntermediateCompensationEventCatching",
    "IntermediateCompensationEventThrowing",
    "EndCompensationEvent"
]
START_EVENT = "StartNoneEvent"
INTERMEDIATE_EVENT = "IntermediateEvent"
END_EVENT = "EndNoneEvent"
UNTYPED_EVENTS = [START_EVENT, INTERMEDIATE_EVENT, END_EVENT]
TERMINATE_EVENT = "EndTermintateEvent"
EVENT_CATEGORIES = (
        UNTYPED_EVENTS + [TERMINATE_EVENT] + MESSAGE_EVENTS + TIMER_EVENTS + ERROR_EVENTS + ESCALATION_EVENTS + SIGNAL_EVENTS + MULTIPLE_EVENTS + CONDITIONAL_EVENTS + LINK_EVENTS + CANCEL_EVENTS + COMPENSATION_EVENTS
)
# Event definitions are modeled as child elements in BPMN XML:
# terminateEventDefinition, messageEventDefinition, timerEventDefinition
EVENT_DEFINITIONS = ["message", "timer", "terminate"]

GATEWAY_CATEGORIES = [
    "Exclusive_Databased_Gateway",
    "ParallelGateway",
    "InclusiveGateway",
    "EventbasedGateway",
    "ComplexGateway"
]
NODE_CATEGORIES = ACTIVITY_CATEGORIES + GATEWAY_CATEGORIES + EVENT_CATEGORIES

DATA_OBJECT = "DataObject"
DATA_STORE = "DataStore"
DATA_MESSAGE = "Message"
DATA_COMMUNICATION = "Communication"
BUSINESS_OBJECT_CATEGORIES = [DATA_OBJECT, DATA_STORE, DATA_MESSAGE]

ARTIFACT_CATEGORIES = ["TextAnnotation", "ITSystem", "Group"]

# bpmndi:BPMNShape
BPMNDI_SHAPE_CATEGORIES = [
    *ACTIVITY_CATEGORIES,
    *EVENT_CATEGORIES,
    *GATEWAY_CATEGORIES,
    *COLLABORATION_CATEGORIES,
    *BUSINESS_OBJECT_CATEGORIES,
    *ARTIFACT_CATEGORIES,
]

# bpmndi:BPMNEdge
# - association is for textAnnotation
MESSAGE_FLOW = "MessageFlow"
SEQUENCE_FLOW = "SequenceFlow"
ASSOCIATION_CATEGORIES = [
    "Association_Bidirectional",
    "Association_Undirected",
    "Association_Unidirectional"
]
CONVERSATION_LINK = "ConversationLink"
BPMNDI_EDGE_CATEGORIES = [SEQUENCE_FLOW, MESSAGE_FLOW, ASSOCIATION_CATEGORIES, CONVERSATION_LINK]

# bpmndi:BPMNLabel
LABEL = "label"
BPMNDI_LABEL_CATEGORIES = [LABEL]

MISC_CATEGORIES = ["gdottedline","gdashedline","gtext","gellipse","gdiamond","grect"]

CATEGORY_GROUPS = {
    "activity": ACTIVITY_CATEGORIES,
    "event": EVENT_CATEGORIES,
    "gateway": GATEWAY_CATEGORIES,
    "collaboration": COLLABORATION_CATEGORIES,
    "business_object": BUSINESS_OBJECT_CATEGORIES,
    "artifact": ARTIFACT_CATEGORIES,
    "label": BPMNDI_LABEL_CATEGORIES,
    "edge": BPMNDI_EDGE_CATEGORIES,
    "misc": MISC_CATEGORIES
}

BPMNDI_SHAPE_GROUPS = [
    "activity",
    "event",
    "gateway",
    "collaboration",
    "business_object",
    "artifact",
]

ALL_CATEGORIES = yamlu.flatten(CATEGORY_GROUPS.values())

EVENT_CATEGORY_TO_NO_POS_TYPE = {
    **{k: "event" for k in UNTYPED_EVENTS},
    **{k: "messageEvent" for k in MESSAGE_EVENTS},
    **{k: "timerEvent" for k in TIMER_EVENTS},
}

def _check_inconsistencies():
    n_bpmndi = (
            len(BPMNDI_SHAPE_CATEGORIES)
            + len(BPMNDI_EDGE_CATEGORIES)
            + len(BPMNDI_LABEL_CATEGORIES)
            + len(MISC_CATEGORIES)
    )
    n_cats = sum(len(g) for g in CATEGORY_GROUPS.values())
    assert n_bpmndi == n_cats, f"{n_bpmndi}, {n_cats}"

    #long_cat_names = set(CATEGORY_TO_LONG_NAME.keys())
    #all_cats = set(yamlu.flatten(CATEGORY_GROUPS.values()))
    #diff = all_cats.difference(long_cat_names)
    #assert len(diff) == 0, diff


#_check_inconsistencies()


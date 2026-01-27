/**
 * Broadcast channel for global events
 */
const broadcastChannel = new window.BroadcastChannel("pub-sub-es");
const isString = (key) => typeof key === "string";
const getEventName = (eventName, caseInsensitive) => {
    if (isString(eventName) && caseInsensitive) {
        return eventName.toLowerCase();
    }
    return eventName;
};
/**
 * Setup subscriber
 */
const createSubscribe = (stack, options) => (event, handler, times = Number.POSITIVE_INFINITY) => {
    const e = getEventName(event, options?.caseInsensitive);
    const listeners = stack[e] || [];
    listeners.push({
        handler,
        times: +times || Number.POSITIVE_INFINITY,
    });
    stack[e] = listeners;
    return { event: e, handler };
};
const isSubscription = (event) => typeof event === "object";
/**
 * Factory function for creating `unsubscribe`
 */
function createUnsubscribe(stack, options) {
    function unsubscribe(eventOrSubscription, handlerOrUndefined) {
        let event;
        let handler;
        if (isSubscription(eventOrSubscription)) {
            handler = eventOrSubscription.handler;
            event = eventOrSubscription.event;
        }
        else {
            event = eventOrSubscription;
            // biome-ignore lint/style/noNonNullAssertion: The function overload defines that if `eventOrSubscription` is not a subscription, `handler` must be defined
            handler = handlerOrUndefined;
        }
        const e = getEventName(event, options?.caseInsensitive);
        const listeners = stack[e];
        if (!listeners) {
            return;
        }
        const idx = listeners.findIndex((listener) => listener.handler === handler);
        if (idx === -1 || idx >= listeners.length) {
            return;
        }
        listeners.splice(idx, 1);
    }
    return unsubscribe;
}
const hasListeners = (listeners) => {
    return Boolean(listeners);
};
/**
 * Factory function for create `publish()`
 */
const createPublish = (stack, options) => {
    const unsubscribe = createUnsubscribe(stack);
    return (...args) => {
        const [event, news, callOptions] = args;
        const eventName = getEventName(event, options?.caseInsensitive);
        const listenersOrUndefined = stack[eventName];
        if (!hasListeners(listenersOrUndefined)) {
            return;
        }
        const listeners = [...listenersOrUndefined];
        for (const listener of listeners) {
            if (--listener.times < 1) {
                unsubscribe(eventName, listener.handler);
            }
        }
        const isAsync = callOptions?.async !== undefined ? callOptions.async : options?.async;
        /**
         * Inform listeners about some news
         */
        const inform = () => {
            for (const listener of listeners) {
                listener.handler(news);
            }
        };
        if (isAsync) {
            setTimeout(inform, 0);
        }
        else {
            inform();
        }
        if (options?.isGlobal && !callOptions?.isNoGlobalBroadcast) {
            try {
                broadcastChannel.postMessage({ event: eventName, news });
            }
            catch (error) {
                if (error instanceof Error && error.name === "DataCloneError") {
                    console.warn(`Could not broadcast '${eventName.toString()}' globally. Payload is not clonable.`);
                }
                else {
                    throw error;
                }
            }
        }
    };
};
function keys(obj) {
    // @ts-expect-error - Object.keys returns the string keys of our type and omits number & symbol but TS's doesn't type the object this way because there are edge cases
    return Object.keys(obj);
}
/**
 * Factory function for creating `clear()`
 */
const createClear = (stack) => () => {
    for (const event of keys(stack)) {
        delete stack[event];
    }
};
/**
 * Create a new empty stack object
 */
const createStack = () => ({});
/**
 * Create a new pub-sub instance
 */
const createPubSub = (options) => {
    const async = Boolean(options?.async);
    const caseInsensitive = Boolean(options?.caseInsensitive);
    const stack = options?.stack || createStack();
    return {
        publish: createPublish(stack, { async, caseInsensitive }),
        subscribe: createSubscribe(stack, { caseInsensitive }),
        unsubscribe: createUnsubscribe(stack, { caseInsensitive }),
        clear: createClear(stack),
        stack,
    };
};
/**
 * Global pub-sub stack object
 */
const globalPubSubStack = createStack();
/**
 * Global pub-sub instance
 */
const globalPubSub = {
    publish: createPublish(globalPubSubStack, { isGlobal: true }),
    subscribe: createSubscribe(globalPubSubStack),
    unsubscribe: createUnsubscribe(globalPubSubStack),
    clear: createClear(globalPubSubStack),
    stack: globalPubSubStack,
};
broadcastChannel.onmessage = ({ data: { event, news } }) => globalPubSub.publish(event, news, { isNoGlobalBroadcast: true });
export { globalPubSub, createPubSub };
export default createPubSub;

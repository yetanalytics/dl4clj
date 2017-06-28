(ns dl4clj.streaming.routes.dl4j-serve-route-builder
  (:import [org.deeplearning4j.streaming.routes DL4jServeRouteBuilder]
           [org.apache.camel.processor DynamicRouter]
           [org.apache.camel.impl DefaultCamelContext]
           [org.apache.camel Processor]))

(defn new-dl4j-serve-route-builder
  ":model-uri (str), uri for a saved multi layer network or computational graph

   :kafka-broker (str), not sure, need to learn more about kafka

   :consuming-topic (str), not sure, need to learn more about kafka

   :comp-graph? (boolean), are we loading a computation graph?

   :output-uri (str), not sure, need to look more into kafka
    - looks like the location to route the result of running data through the supplied model

   :final-processor (Camel Processor), believe performns the processing/routing of the request?

   :group-id (str), the group id for a kafka stream?, defaults to 'dl4j-serving'

   :zoo-keeper-host (str), the path to the zoo keeper host, defaults to 'localhost'

   :zoo-keeper-port (int), the port zoo keeper lives on, defaults to 2181

   :before-processor (Camel Processor), has some interaction with the processing/routing
    - defaults to nil, as this creates a new processor in the dl4j java src

   :configure? (boolean), do you want to configure the route builder or just return
    the object itself without the configure call made"
  [& {:keys [model-uri kafka-broker consuming-topic
             comp-graph? output-uri final-processor
             group-id zoo-keeper-host zoo-keeper-port
             before-processor configure?]
      :or {comp-graph? false
           final-processor (DynamicRouter. (DefaultCamelContext.))
           before-processor nil
           zoo-keeper-host "localhost"
           zoo-keeper-port 2181
           group-id "dl4j-serving"
           configure? true}}]
  (let [route-builder (DL4jServeRouteBuilder. model-uri kafka-broker consuming-topic
                                              comp-graph? output-uri final-processor
                                              group-id zoo-keeper-host zoo-keeper-port before-processor)]
    (if configure?
      (doto  route-builder
        .configure)
      route-builder)))

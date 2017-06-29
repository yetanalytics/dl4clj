(ns dl4clj.streaming.kafka.ndarray-publisher
  (:import [org.deeplearning4j.streaming.kafka NDArrayPublisher]
           [org.apache.camel.impl DefaultCamelContext DefaultProducerTemplate]
           [org.nd4j.linalg.api.ndarray INDArray])
  (:require [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]
            [dl4clj.utils :refer [array-of]]))

(defn new-ndarray-kafka-publisher
  [& {:keys [kafka-uri kafka-topic]}]
  (let [camel-context (DefaultCamelContext.)
        producer-template (DefaultProducerTemplate. camel-context)]
    (NDArrayPublisher. camel-context kafka-topic kafka-uri producer-template false)))

(defn start-publisher!
  "starts the publisher"
  [publisher]
  (doto publisher (.start)))

(defn publish!
  "publish an INDArray, can be a single array or an array of arrays
   - when an array of arrays, data must be a collection of INDarrays and many-arrays? = true"
  [& {:keys [data publisher many-arrays?]}]
  (if many-arrays?
    (doto publisher (.publish (array-of :data data
                                        :java-type INDArray)))
    (doto publisher (.publish (vec-or-matrix->indarray data)))))

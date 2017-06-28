(ns dl4clj.streaming.kafka.ndarray-publisher
  (:import [org.deeplearning4j.streaming.kafka NDArrayPublisher]
           [org.apache.camel.impl DefaultCamelContext DefaultProducerTemplate]))

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
  "publish an INDArray, can be a single array or an array of arrays"
  [& {:keys [ind-array publisher]}]
  (doto publisher (.publish ind-array)))

(ns ^{:doc "see: https://deeplearning4j.org/doc/org/deeplearning4j/streaming/routes/CamelKafkaRouteBuilder.html"}
    dl4clj.streaming.routes.camel-kafka-route-builder
  (:import [org.deeplearning4j.streaming.routes CamelKafkaRouteBuilder CamelKafkaRouteBuilder$Builder]))

;; I need to look into camel and kafka to understand whats going on here

(defn c-k-route-builder
  "kafka route builder via camel.

  :builder (builder), this key provides the option to supply an existing CamelKafkaRoutebuilder
   - if not supplied, defaults to a fresh instance of the class

  :build? (boolean), do you want to build the builder?, defaults to true

  :camel-context (), dont have a good desc yet
   type = org.apache.camel.CamelContext

  :data-type-unmarshal (str), dont have a good desc

  :datavec-marshaller (str), dont have a good desc

  :input-format (str), dont have a good desc

  :input-uri (str), the uri for the kafka input stream (best guess)

  :kafka-broker-list (string), dont have a good desc

  :camel-processor (), dont have a good desc
   type = org.apache.camel.Processor

  :topic-name (str), something to do with kafka

  :writable-converter (str), dont have a good desc

  :zoo-keeper-host (str), the zoo-keeper host to use

  :zoo-keeper-port (int), the port to talk to zoo-keeper on"
  [& {:keys [builder build? camel-context data-type-unmarshal
             datavec-marshaller input-format input-uri
             kafka-broker-list camel-processor
             topic-name writable-converter zoo-keeper-host
             zoo-keeper-port]
      :or {builder (CamelKafkaRouteBuilder$Builder.)
           build? true}
      :as opts}]
  (cond-> builder
    (contains? opts :camel-context) (.camelContext camel-context)
    (contains? opts :data-type-unmarshal) (.dataTypeUnMarshal data-type-unmarshal)
    (contains? opts :datavec-marshaller) (.datavecMarshaller datavec-marshaller)
    (contains? opts :input-format) (.inputFormat input-format)
    (contains? opts :input-uri) (.inputUri input-uri)
    (contains? opts :kafka-broker-list) (.kafkaBrokerList kafka-broker-list)
    (contains? opts :camel-processor) (.processor camel-processor)
    (contains? opts :topic-name) (.topicName topic-name)
    (contains? opts :writable-converter) (.writableConverter writable-converter)
    (contains? opts :zoo-keeper-host) (.zooKeeperHost zoo-keeper-host)
    (contains? opts :zoo-keeper-port) (.zooKeeperPort zoo-keeper-port)
    (true? build?) (.build)))

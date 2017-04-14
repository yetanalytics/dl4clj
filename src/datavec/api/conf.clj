(ns dl4clj.datavec.api.conf
  (:import [org.datavec.api.conf Configuration$IntegerRanges
            Configuration Configured]
           [org.datavec.api.io.serializers SerializationFactory])
  (:require [clojure.string :as s]))

;;https://deeplearning4j.org/datavecdoc/org/datavec/api/conf/Configuration.html
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/conf/Configured.html
(SerializationFactory. (Configuration.))
;;https://deeplearning4j.org/datavecdoc/org/datavec/api/io/serializers/SerializationFactory.html
;; constructor takes a configuration as the only arg

(defn new-config
  [& {:keys [load-defaults? other-conf]
      :as opts}]
  (cond
    (contains? opts :load-defaults?) (Configuration. load-defaults?)
    (contains? opts :other-conf) (Configuration. other-conf)
    :else
    (Configuration.)))

(defn configured
  [& {:keys [conf]
      :as opts}]
  (if (contains? opts :conf)
    (Configured. conf)
    (Configured.)))
;; has these methods: getConf, setConf

(defn serialization-factory [conf]
  (SerializationFactory. conf))
;;3 methods: getDeserializer, getSerialization, getSerializer

(defn integer-ranges
  "This fn constructs the string Integerranges constructor is expecting,
  for ranges, pass a string of {upper}-{lower} either uppor or lower can be omited
  for a single value, pass a string of the int

  ie. 2-3,5,7- is the output of this fn when passed '2-3' 5 '7- and means
  2, 3, 5, and 7, 8, 9,

  NOTE: this works for positive integers only"
  [& {:keys [ranges]}]
  (let [ranges-with-commas (loop [result []
                                  r ranges]
                             (if (empty? r)
                               result
                               (let [data (first r)]
                                 (if (= 1 (count r))
                                   (recur
                                    (conj result (str data))
                                    (rest r))
                                   (recur
                                  (conj result (str data ","))
                                  (rest r))))))
        all-together (s/join ranges-with-commas)]
    all-together))

(defn integer-ranges-parser
  "ranges should be a collection of ranges formatted according to the doc string
  of integer-ranges"
  [& {:keys [ranges]
      :as opts}]
  (if (contains? opts :ranges)
    (Configuration$IntegerRanges. (integer-ranges :ranges ranges))
    (Configuration$IntegerRanges.)))

(defn is-included-in-range?
  [& {:keys [this included]}]
  (.isIncluded this included))

(defn config-methods
  ;; dealing with hadoop
  [conf & {:keys [add-default-resource add-resource clear?
                  dump-config-writer get-property-by-name]
           :as opts}]
  (cond-> conf
    (contains? opts :add-default-resource) (.addDefaultResource add-default-resource)
    (contains? opts :add-resource) (.addResource add-resource)
    (and (contains? opts :clear?)
         (true? clear?)) (.clear)
    (contains? opts :dump-config-writer) (.dumpConfiguration dump-config-writer)
    (contains? opts :get-property-by-name) (.get get-property-by-name)
    ;; come back and finish when there is a need for this
    ;; https://deeplearning4j.org/datavecdoc/org/datavec/api/conf/Configuration.html
    ))

(defn serialization-methods
  [conf & {:keys [java-class conf-opts
                  get-deserializer? get-serialization?
                  get-serializer? get-conf? set-conf?]
           :as opts}]
  (let [this (serialization-factory conf)]
   (cond-> this
    (and (contains? opts :get-deserializer?)
         (true? get-deserializer?)) (.getDeserializer java-class)
    (and (contains? opts :get-serialization?)
         (true? get-serialization?)) (.getSerialization java-class)
    (and (contains? opts :get-serializer?)
         (true? get-serializer?)) (.getSerializer java-class)
    (and (contains? opts :get-conf?)
         (true? get-conf?)) (.getConf )
    (and (contains? opts :set-conf?)
         (true? set-conf?)
         (contains? opts :conf-opts)) (.setConf conf-opts))))

(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/BackpropType.html"}
  dl4clj.nn.conf.backprop-type
  (require [dl4clj.utils :refer (camel-to-dashed)])
  (:import [org.deeplearning4j.nn.conf BackpropType]))

(defn value-of [k]
  (let [k (keyword (clojure.string/lower-case (camel-to-dashed (name k))))]
    (condp = k
      :standard (BackpropType/valueOf "Standard")
      :truncated-bptt (BackpropType/valueOf "TruncatedBPTT")    
      (BackpropType/valueOf (clojure.string/replace (clojure.string/upper-case (name k)) "-" "_")))))

(defn values []
  [:standard :truncated-bptt])

(comment

  (values)
  (value-of :truncated-bptt)
  (value-of :foo)
  
)

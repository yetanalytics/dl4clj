(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/weights/WeightInit.html"}
  dl4clj.nn.weights.weight-init
  (:import [org.deeplearning4j.nn.weights WeightInit]))

(defn value-of [k]
  (if (string? k)
    (WeightInit/valueOf k)
    (WeightInit/valueOf (clojure.string/upper-case (name k)))))

(defn values []
  (map #(keyword (clojure.string/lower-case %)) (WeightInit/values)))

(comment

  (values)
  (value-of :normalized)
  (value-of "NORMALIZED")
  (value-of :foo)

)

(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Model.html"}
  dl4clj.nn.api.model
  (:import [org.deeplearning4j.nn.api Model]))

(defmulti num-params (fn [x & more] (type x)))

(defmulti fit (fn [x & more] (type x)))

;; i.e.:
;; Interface `A` has methods `m(x)`, `m(x,y)`, ...
;; Interface `B` has methods `m(u)`, `m(u,v)`, ...
;; Class `C` implements both interfaces `A` and `B`
;; The question is how to expose all this in clojure?



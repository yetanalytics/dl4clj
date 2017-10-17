(ns dl4clj.nn.api.gradient
  (:import [org.deeplearning4j.nn.gradient DefaultGradient])
  (:require [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]
            [dl4clj.utils :refer [obj-or-code?]]
            [clojure.core.match :refer [match]]))

(defn clear!
  "Clear residual parameters (useful for returning a gradient and then clearing old objects)"
  [grad & {:keys [as-code?]
           :or {as-code? true}}]
  (match [grad]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~grad .clear))
         :else
         (doto grad .clear)))

(defn flattening-order-for-variables
  "Return the gradient flattening order for the specified variable, or null if it is not explicitly set"
  [& {:keys [grad variable as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:grad (_ :guard seq?)
           :variable (:or (_ :guard seq?)
                          (_ :guard string?))}]
         (obj-or-code? as-code? `(.flatteningOrderForVariable ~grad ~variable))
         :else
         (.flatteningOrderForVariable grad variable)))

(defn get-gradient-for
  "The gradient for the given variable"
  [& {:keys [grad variable as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:grad (_ :guard seq?)
           :variable (:or (_ :guard seq?)
                          (_ :guard string?))}]
         (obj-or-code? as-code? `(.getGradientFor ~grad ~variable))
         :else
         (.getGradientFor grad variable)))

(defn gradient
  "The full gradient as one flat vector"
  [& {:keys [grad order as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:grad (_ :guard seq?) :order (_ :guard seq?)}]
         (obj-or-code? as-code? `(.gradient ~grad ~order))
         [{:grad _ :order _}]
         (.gradient grad order)
         [{:grad (_ :guard seq?)}]
         (obj-or-code? as-code? `(.gradient ~grad))
         :else
         (.gradient grad)))

(defn gradient-for-variable
  "Gradient look up table"
  [grad & {:keys [as-code?]
           :or {as-code? true}}]
  (match [grad]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.gradientForVariable ~grad))
         :else
         (.gradientForVariable grad)))

(defn set-gradient-for!
  "Update gradient for the given variable; also (optionally) specify the order in which the array should be flattened to a row vector"
  [& {:keys [grad variable new-gradient flattening-order as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:grad (_ :guard seq?)
           :variable (:or (_ :guard seq?)
                          (_ :guard string?))
           :new-gradient (:or (_ :guard vector?)
                              (_ :guard seq?))
           :flattening-order _}]
         (obj-or-code?
          as-code?
          `(doto ~grad
             (.setGradientFor ~variable (vec-or-matrix->indarray ~new-gradient)
                              ~flattening-order)))
         [{:grad _
           :variable _
           :new-gradient _
           :flattening-order _}]
         (doto grad (.setGradientFor variable
                                     (vec-or-matrix->indarray new-gradient)
                                     flattening-order))
         [{:grad (_ :guard seq?)
           :variable (:or (_ :guard seq?)
                          (_ :guard string?))
           :new-gradient (:or (_ :guard vector?)
                              (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~grad
            (.setGradientFor ~variable (vec-or-matrix->indarray ~new-gradient))))
         :else
         (doto grad (.setGradientFor variable (vec-or-matrix->indarray new-gradient)))))

(ns ^{:doc "implementation of the Convex Optimizer interface
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/ConvexOptimizer.html"}
    dl4clj.optimize.api.convex-optimizer
  (:import [org.deeplearning4j.optimize.api ConvexOptimizer])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [obj-or-code?]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; getters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-batch-size
  "returns the batch size for the optimizer"
  [& {:keys [optim as-code?]
      :or {as-code? true}}]
  (match [optim]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.batchSize ~optim))
         :else
         (.batchSize optim)))

(defn get-conf
  "get the nn-conf associated with this optimizer"
  [& {:keys [optim as-code?]
      :or {as-code? true}}]
  (match [optim]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getConf ~optim))
         :else
         (.getConf optim)))

(defn get-updater
  "returns the updater associated with this optimizer"
  [& {:keys [optim as-code?]
      :or {as-code? true}}]
  (match [optim]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getUpdater ~optim))
         :else
         (.getUpdater optim)))

(defn get-gradient-and-score
  "the gradient and score for this optimizer"
  [& {:keys [optim as-code?]
      :or {as-code? true}}]
  (match [optim]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.gradientAndScore ~optim))
         :else
         (.gradientAndScore optim)))

(defn get-score
  "The score for the optimizer so far"
  [& {:keys [optim as-code?]
      :or {as-code? true}}]
  (match [optim]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.score ~optim))
         :else
         (.score optim)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; setters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn set-batch-size!
  "set the batch size for the optimizer

  :batch-size (int), the batch size

  returns the optimizer"
  [& {:keys [optim batch-size as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:optim (_ :guard seq?)
           :batch-size (:or (_ :guard number?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~optim (.setBatchSize (int ~batch-size))))
         :else
         (doto optim (.setBatchSize batch-size))))

(defn set-listeners!
  "sets the listeners for the supplied optimizer

  :listeners (collection), a collection of listeners
   - clojure data structures can be used

  returns the optimizer"
  [& {:keys [optim listeners as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:optim (_ :guard seq?)
           :listeners (:or (_ :guard coll?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~optim (.setListeners ~listeners)))
         :else
         (doto optim (.setListeners listeners))))

(defn set-updater!
  "sets the updater for the optimizer

  :updater (updater), an updater to add to the optimizer

  returns the optimizer"
  [& {:keys [optim updater as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:optim (_ :guard seq?)
           :updater (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~optim (.setUpdater ~updater)))
         :else
         (doto optim (.setUpdater updater))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; misc
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn optimize
  "calls optimize! returns a boolean"
  [& {:keys [optim as-code?]
      :or {as-code? true}}]
  (match [optim]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.optimize ~optim))
         :else
         (.optimize optim)))

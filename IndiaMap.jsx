import React from "react";
import { ComposableMap, Geographies, Geography } from "react-simple-maps";
import indiaGeoUrl from "../assets/india-states.json";

export default function IndiaMap({ highlightState }) {
    // normalize highlightState once (if provided)
    const normalizedHighlight = highlightState
        ? String(highlightState).trim().toLowerCase()
        : "";

    return (
        <div className="bg-white p-4 shadow rounded-lg">
            <h3 className="text-lg font-semibold mb-2">India Map (Highlight by District)</h3>

            <ComposableMap
                projection="geoMercator"
                projectionConfig={{
                    center: [78.9629, 22.5937], // longitude, latitude of India
                    scale: 700, // zoom level (try 800–1200)
                }}
                width={400}
                height={400}
            >
                <Geographies geography={indiaGeoUrl}>
                    {({ geographies }) =>
                        geographies.map((geo) => {
                            const rawName =
                                geo?.properties?.st_nm ??
                                geo?.properties?.NAME_1 ??
                                geo?.properties?.name ??
                                "";

                            const stateName = String(rawName).trim().toLowerCase();
                            const isHighlighted =
                                highlightState &&
                                stateName === String(highlightState).trim().toLowerCase();

                            return (
                                <Geography
                                    key={geo.rsmKey}
                                    geography={geo}
                                    fill={isHighlighted ? "#2563eb" : "#E5E7EB"}
                                    stroke="#fff"
                                    strokeWidth={0.5}
                                />
                            );
                        })
                    }
                </Geographies>
            </ComposableMap>
        </div>
    );
}

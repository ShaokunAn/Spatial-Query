function createPlugins(utilsForPlugins) {
    const {
        React,
        PluginFileType,
        PluginViewType,
        PluginCoordinationType,
        PluginJointFileType,
        z,
        useCoordination,
    } = utilsForPlugins;
    const e = React.createElement;
    function SpatialQueryView(props) {
        const { coordinationScopes } = props;
        const [{
            queryParams,
            obsSetSelection,
        }, {
            setQueryParams,
        }] = useCoordination(['queryParams', 'obsSetSelection', 'obsType'], coordinationScopes);

        const [uuid, setUuid] = React.useState(1);
        const [queryType, setQueryType] = React.useState('grid');
        const [maxDist, setMaxDist] = React.useState(100);
        const [minSize, setMinSize] = React.useState(4);
        const [minCount, setMinCount] = React.useState(10);
        const [minSupport, setMinSupport] = React.useState(0.5);

        const cellTypeOfInterest = obsSetSelection?.length === 1 && obsSetSelection[0][0] === "Cell Type"
            ? obsSetSelection[0][1]
            : null;

        const onQueryTypeChange = React.useCallback((e) => {
            setQueryType(e.target.value);
        }, []);

        return e('div', { className: "spatial-query" }, [
            e('p', {}, 'Spatial Query Manager'),
            e('label', {}, [
                'Query type ',
                e('select', { onChange: onQueryTypeChange }, [
                    e('option', { value: 'grid' }, 'Grid-based'),
                    e('option', { value: 'rand' }, 'Random-based'),
                    e('option', { value: 'ct-center', disabled: cellTypeOfInterest === null }, 'Cell type of interest'),
                ]),
            ]),
            e('br'),
            e('label', {}, [
                // Maximum distance to consider a cell as a neighbor.
                'Max. Dist. ',
                e('input', { type: 'range', value: maxDist, onChange: e => setMaxDist(parseFloat(e.target.value)), min: 50, max: 250, step: 1 }),
                maxDist,
            ]),
            e('br'),
            e('label', {}, [
                // Minimum neighborhood size for each point to consider.
                'Min. Size ',
                e('input', { type: 'range', value: minSize, onChange: e => setMinSize(parseFloat(e.target.value)), min: 0, max: 20, step: 1 }),
                minSize,
            ]),
            e('br'),
            e('label', {}, [
                // Minimum number of cell type to consider.
                'Min. Count ',
                e('input', { type: 'range', value: minCount, onChange: e => setMinCount(parseFloat(e.target.value)), min: 0, max: 30, step: 1 }),
                minCount,
            ]),
            e('br'),
            e('label', {}, [
                // Threshold of frequency to consider a pattern as a frequent pattern.
                'Min. Support ',
                e('input', { type: 'range', value: minSupport, onChange: e => setMinSupport(parseFloat(e.target.value)), min: 0, max: 1, step: 0.01 }),
                minSupport,
            ]),
            e('br'),
            // TODO: disDuplicates: Distinguish duplicates in patterns.
            e('button', { onClick: (e) => {
                setQueryParams({
                    cellTypeOfInterest,
                    queryType,
                    maxDist,
                    minSize,
                    minCount,
                    minSupport,
                    uuid,
                });
                setUuid(uuid+1);
            } }, 'Find patterns'),
        ]);
    }

    const pluginCoordinationTypes = [
        new PluginCoordinationType('queryParams', null, z.object({
            cellTypeOfInterest: z.string().nullable(),
            queryType: z.enum(['grid', 'rand', 'ct-center']),
            maxDist: z.number(),
            minSize: z.number(),
            minCount: z.number(),
            minSupport: z.number(),
            disDuplicates: z.boolean(),
            uuid: z.number(),
        }).partial().nullable()),
    ];

    const pluginViewTypes = [
        new PluginViewType('spatialQuery', SpatialQueryView, ['queryParams', 'obsSetSelection', 'obsType']),
    ];
    return { pluginViewTypes, pluginCoordinationTypes };
}
export default { createPlugins };
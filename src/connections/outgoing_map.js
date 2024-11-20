function mapFromDittoProtocolMsg(namespace, id, group, channel, criterion, action, path, dittoHeaders, value, status, extra){
	let payload = { position: value.gps.properties.position, orientation: value.gps.properties.orientation, thingId: 'test' };
	let bytePayload = null;
	let contentType = 'application/json';
	let topic = 'devices/out/' + namespace + ':' + id;
	return Ditto.buildExternalMsg( 
	dittoHeaders,
	JSON.stringify(payload),
	bytePayload,
	contentType);
}
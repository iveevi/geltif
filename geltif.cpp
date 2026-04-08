#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <tuple>
#include <unordered_map>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "geltif.hpp"

static std::string load_texture(const cgltf_texture_view &view, const std::filesystem::path &directory, Scene &scene)
{
	if (!view.texture || !view.texture->image)
		return {};

	const cgltf_image *image = view.texture->image;

	int w, h, channels;
	uint8_t *pixels = nullptr;
	std::string key;

	if (image->uri) {
		key = directory / image->uri;
		if (scene.textures.contains(key))
			return key;
		pixels = stbi_load(key.c_str(), &w, &h, &channels, 4);
	} else if (image->buffer_view) {
		key = "__embedded_" + std::to_string(reinterpret_cast <uintptr_t> (image->buffer_view));
		if (scene.textures.contains(key))
			return key;
		const uint8_t *raw = static_cast <const uint8_t *> (image->buffer_view->buffer->data) + image->buffer_view->offset;
		pixels = stbi_load_from_memory(raw, static_cast <int> (image->buffer_view->size), &w, &h, &channels, 4);
	}

	if (!pixels)
		return {};

	RawTexture img;
	img.width = static_cast <uint32_t> (w);
	img.height = static_cast <uint32_t> (h);
	img.pixels.assign(pixels, pixels + w * h * 4);
	stbi_image_free(pixels);
	scene.textures.emplace(key, std::move(img));
	return key;
}

void Geometry::recalculate_normals()
{
	normals.assign(positions.size(), glm::vec3(0.0f));

	for (const auto &tri : triangles) {
		auto v1 = positions[tri.x];
		auto v2 = positions[tri.y];
		auto v3 = positions[tri.z];
		auto n = glm::cross(v2 - v1, v3 - v1);
		normals[tri.x] += n;
		normals[tri.y] += n;
		normals[tri.z] += n;
	}

	for (auto &n : normals) {
		if (glm::length(n) > 1e-6f)
			n = glm::normalize(n);
	}
}

void Geometry::deduplicate()
{
	using Key = std::tuple <glm::vec3, glm::vec3, glm::vec2>;

	struct KeyHash {
		size_t operator()(const Key &k) const {
			auto combine = [](size_t a, size_t b) {
				return a ^ (b * 2654435769ull + (a << 6) + (a >> 2));
			};
			auto hf = std::hash <float> {};
			const auto &[p, n, uv] = k;
			size_t s = 0;
			for (float f : { p.x, p.y, p.z, n.x, n.y, n.z, uv.x, uv.y })
				s = combine(s, hf(f));
			return s;
		}
	};

	std::unordered_map <Key, uint32_t, KeyHash> seen;
	std::vector <glm::vec3> new_pos, new_nor;
	std::vector <glm::vec2> new_uv;
	std::vector <glm::uvec3> new_tris;

	for (const auto &tri : triangles) {
		glm::uvec3 new_tri {};
		for (int k = 0; k < 3; k++) {
			uint32_t idx = (&tri.x)[k];
			glm::vec3 p = idx < positions.size() ? positions[idx] : glm::vec3(0);
			glm::vec3 n = idx < normals.size() ? normals[idx] : glm::vec3(0);
			glm::vec2 uv = idx < uvs.size() ? uvs[idx] : glm::vec2(0);
			auto key = std::make_tuple(p, n, uv);
			auto it = seen.find(key);
			if (it == seen.end()) {
				uint32_t ni = static_cast <uint32_t> (new_pos.size());
				seen[key] = ni;
				new_pos.push_back(p);
				new_nor.push_back(n);
				new_uv.push_back(uv);
				(&new_tri.x)[k] = ni;
			} else {
				(&new_tri.x)[k] = it->second;
			}
		}
		new_tris.push_back(new_tri);
	}

	positions = std::move(new_pos);
	normals = std::move(new_nor);
	uvs = std::move(new_uv);
	triangles = std::move(new_tris);
}

static glm::vec4 sample_channel(const AnimationChannel &ch, float t)
{
	if (ch.times.size() == 1 || t <= ch.times.front())
		return ch.values.front();
	if (t >= ch.times.back())
		return ch.values.back();

	auto it = std::upper_bound(ch.times.begin(), ch.times.end(), t);
	size_t i = static_cast <size_t> (it - ch.times.begin()) - 1;

	float t0 = ch.times[i], t1 = ch.times[i + 1];
	float alpha = (t - t0) / (t1 - t0);

	return glm::mix(ch.values[i], ch.values[i + 1], alpha);
}

static glm::quat sample_rotation(const AnimationChannel &ch, float t)
{
	auto to_quat = [](const glm::vec4 &v) {
		return glm::quat(v.w, v.x, v.y, v.z);
	};

	if (ch.times.size() == 1 || t <= ch.times.front())
		return to_quat(ch.values.front());
	if (t >= ch.times.back())
		return to_quat(ch.values.back());

	auto it = std::upper_bound(ch.times.begin(), ch.times.end(), t);
	size_t i = static_cast <size_t> (it - ch.times.begin()) - 1;

	float t0 = ch.times[i], t1 = ch.times[i + 1];
	float alpha = (t - t0) / (t1 - t0);

	return glm::slerp(to_quat(ch.values[i]), to_quat(ch.values[i + 1]), alpha);
}

void Animation::advance(float dt)
{
	if (duration <= 0.0f)
		return;
	current_time = std::fmod(current_time + dt, duration);
}

glm::mat4 Animation::evaluate() const
{
	glm::vec3 T = rest_translation;
	glm::quat R = rest_rotation;
	glm::vec3 S = rest_scale;

	if (!translation.times.empty())
		T = glm::vec3(sample_channel(translation, current_time));
	if (!rotation.times.empty())
		R = sample_rotation(rotation, current_time);
	if (!scale.times.empty())
		S = glm::vec3(sample_channel(scale, current_time));

	return glm::translate(glm::mat4(1.0f), T) * glm::mat4_cast(R) * glm::scale(glm::mat4(1.0f), S);
}

void Camera::update(float dt)
{
	if (!animation)
		return;
	animation->advance(dt);
	transform = parent_transform * animation->evaluate();
}

std::expected <Scene, std::string> Scene::from_file(const std::filesystem::path &path)
{
	cgltf_options options {};
	cgltf_data *data = nullptr;

	cgltf_result result = cgltf_parse_file(&options, path.c_str(), &data);
	if (result != cgltf_result_success)
		return std::unexpected("failed to parse glTF: " + path.string());

	result = cgltf_load_buffers(&options, data, path.c_str());
	if (result != cgltf_result_success) {
		cgltf_free(data);
		return std::unexpected("failed to load glTF buffers: " + path.string());
	}

	result = cgltf_validate(data);
	if (result != cgltf_result_success) {
		cgltf_free(data);
		return std::unexpected("glTF validation failed: " + path.string());
	}

	Scene scene;
	auto directory = path.parent_path();

	// Extract materials
	for (cgltf_size i = 0; i < data->materials_count; i++) {
		const cgltf_material &mat = data->materials[i];
		Material material;

		if (mat.has_pbr_metallic_roughness) {
			const auto &pbr = mat.pbr_metallic_roughness;
			material.albedo = load_texture(pbr.base_color_texture, directory, scene);
			material.metallic_roughness = load_texture(pbr.metallic_roughness_texture, directory, scene);
			material.base_color = glm::vec3(pbr.base_color_factor[0], pbr.base_color_factor[1], pbr.base_color_factor[2]);
			material.metallic_factor = pbr.metallic_factor;
			material.roughness_factor = pbr.roughness_factor;
		}

		material.normals = load_texture(mat.normal_texture, directory, scene);
		material.occlusion = load_texture(mat.occlusion_texture, directory, scene);
		material.emissive = load_texture(mat.emissive_texture, directory, scene);
		material.emissive_factor = glm::vec3(mat.emissive_factor[0], mat.emissive_factor[1], mat.emissive_factor[2]);
		if (mat.has_emissive_strength)
			material.emissive_strength = mat.emissive_strength.emissive_strength;
		material.alpha_cutoff = mat.alpha_cutoff;

		scene.materials.push_back(std::move(material));
	}

	// Ensure at least one default material for primitives with no material
	if (scene.materials.empty())
		scene.materials.emplace_back();

	// Walk nodes for meshes, lights, cameras
	std::unordered_map <const cgltf_node *, size_t> camera_node_map;
	for (cgltf_size ni = 0; ni < data->nodes_count; ni++) {
		const cgltf_node &node = data->nodes[ni];

		cgltf_float world[16];
		cgltf_node_transform_world(&node, world);
		glm::mat4 transform = glm::make_mat4(world);

		if (node.mesh) {
			for (cgltf_size pi = 0; pi < node.mesh->primitives_count; pi++) {
				const cgltf_primitive &prim = node.mesh->primitives[pi];
				if (prim.type != cgltf_primitive_type_triangles)
					continue;

				Geometry geo;

				const cgltf_accessor *pos_acc = cgltf_find_accessor(&prim, cgltf_attribute_type_position, 0);
				if (pos_acc) {
					std::vector <float> buf(pos_acc->count * 3);
					cgltf_accessor_unpack_floats(pos_acc, buf.data(), buf.size());
					geo.positions.resize(pos_acc->count);
					std::memcpy(geo.positions.data(), buf.data(), buf.size() * sizeof(float));
				}

				const cgltf_accessor *nor_acc = cgltf_find_accessor(&prim, cgltf_attribute_type_normal, 0);
				if (nor_acc) {
					std::vector <float> buf(nor_acc->count * 3);
					cgltf_accessor_unpack_floats(nor_acc, buf.data(), buf.size());
					geo.normals.resize(nor_acc->count);
					std::memcpy(geo.normals.data(), buf.data(), buf.size() * sizeof(float));
				}

				const cgltf_accessor *uv_acc = cgltf_find_accessor(&prim, cgltf_attribute_type_texcoord, 0);
				if (uv_acc) {
					std::vector <float> buf(uv_acc->count * 2);
					cgltf_accessor_unpack_floats(uv_acc, buf.data(), buf.size());
					geo.uvs.resize(uv_acc->count);
					std::memcpy(geo.uvs.data(), buf.data(), buf.size() * sizeof(float));
				} else {
					geo.uvs.assign(geo.positions.size(), glm::vec2(0.0f));
				}

				if (prim.indices) {
					std::vector <uint32_t> indices(prim.indices->count);
					cgltf_accessor_unpack_indices(prim.indices, indices.data(), sizeof(uint32_t), indices.size());
					geo.triangles.resize(indices.size() / 3);
					std::memcpy(geo.triangles.data(), indices.data(), indices.size() * sizeof(uint32_t));
				}

				uint32_t mat_idx = 0;
				if (prim.material)
					mat_idx = static_cast <uint32_t> (prim.material - data->materials);

				uint32_t geo_idx = static_cast <uint32_t> (scene.geometries.size());
				scene.geometries.push_back(std::move(geo));
				scene.entities.push_back(Entity {
					.geometry_index = geo_idx,
					.material_index = mat_idx,
					.transform = transform,
				});
			}
		}

		if (node.light) {
			const cgltf_light &cl = *node.light;

			auto color = glm::vec3(cl.color[0], cl.color[1], cl.color[2]);
			auto position = glm::vec3(transform[3]);
			auto direction = glm::normalize(glm::vec3(transform * glm::vec4(0, 0, -1, 0)));

			switch (cl.type) {
			case cgltf_light_type_directional:
				// glTF exports directional light intensity in lux;
				// convert back to W/m² to match Blender's displayed value
				scene.lights.push_back(DirectionalLight {
					.color = color,
					.intensity = cl.intensity / 683.0f,
					.direction = direction,
				});
				break;
			case cgltf_light_type_point:
				scene.lights.push_back(PointLight {
					.color = color,
					.intensity = cl.intensity,
					.position = position,
					.range = cl.range,
				});
				break;
			case cgltf_light_type_spot:
				scene.lights.push_back(SpotLight {
					.color = color,
					.intensity = cl.intensity,
					.position = position,
					.direction = direction,
					.range = cl.range,
					.inner_cone = cl.spot_inner_cone_angle,
					.outer_cone = cl.spot_outer_cone_angle,
				});
				break;
			default:
				continue;
			}
		}

		if (node.camera) {
			const cgltf_camera &cc = *node.camera;
			Camera cam;
			cam.transform = transform;

			if (node.parent) {
				cgltf_float parent_world[16];
				cgltf_node_transform_world(node.parent, parent_world);
				cam.parent_transform = glm::make_mat4(parent_world);
			}

			if (cc.type == cgltf_camera_type_perspective) {
				cam.type = Camera::Type::ePerspective;
				cam.fov = cc.data.perspective.yfov;
				cam.near_plane = cc.data.perspective.znear;
				if (cc.data.perspective.has_aspect_ratio)
					cam.aspect = cc.data.perspective.aspect_ratio;
				if (cc.data.perspective.has_zfar)
					cam.far_plane = cc.data.perspective.zfar;
			} else if (cc.type == cgltf_camera_type_orthographic) {
				cam.type = Camera::Type::eOrthographic;
				cam.xmag = cc.data.orthographic.xmag;
				cam.ymag = cc.data.orthographic.ymag;
				cam.near_plane = cc.data.orthographic.znear;
				cam.far_plane = cc.data.orthographic.zfar;
			}

			scene.cameras.push_back(cam);
			camera_node_map[&node] = scene.cameras.size() - 1;
		}
	}

	// Parse animations targeting camera nodes
	for (cgltf_size ai = 0; ai < data->animations_count; ai++) {
		const cgltf_animation &anim = data->animations[ai];

		for (cgltf_size ci = 0; ci < anim.channels_count; ci++) {
			const cgltf_animation_channel &channel = anim.channels[ci];

			if (!channel.target_node)
				continue;
			auto it = camera_node_map.find(channel.target_node);
			if (it == camera_node_map.end())
				continue;

			auto &cam = scene.cameras[it->second];
			if (!cam.animation) {
				cam.animation.emplace();

				// Extract the node's local rest-pose TRS
				const cgltf_node *n = channel.target_node;
				if (n->has_translation)
					cam.animation->rest_translation = glm::vec3(n->translation[0], n->translation[1], n->translation[2]);
				if (n->has_rotation)
					cam.animation->rest_rotation = glm::quat(n->rotation[3], n->rotation[0], n->rotation[1], n->rotation[2]);
				if (n->has_scale)
					cam.animation->rest_scale = glm::vec3(n->scale[0], n->scale[1], n->scale[2]);
			}

			const cgltf_animation_sampler &sampler = *channel.sampler;

			std::vector <float> times(sampler.input->count);
			cgltf_accessor_unpack_floats(sampler.input, times.data(), times.size());

			size_t components = 0;
			AnimationChannel *target = nullptr;

			switch (channel.target_path) {
			case cgltf_animation_path_type_translation:
				components = 3;
				target = &cam.animation->translation;
				break;
			case cgltf_animation_path_type_rotation:
				components = 4;
				target = &cam.animation->rotation;
				break;
			case cgltf_animation_path_type_scale:
				components = 3;
				target = &cam.animation->scale;
				break;
			default: continue;
			}

			std::vector <float> raw(sampler.output->count * components);
			cgltf_accessor_unpack_floats(sampler.output, raw.data(), raw.size());

			target->times = std::move(times);
			target->values.resize(sampler.output->count);
			for (size_t k = 0; k < sampler.output->count; k++) {
				if (components == 4)
					target->values[k] = glm::vec4(raw[k * 4], raw[k * 4 + 1], raw[k * 4 + 2], raw[k * 4 + 3]);
				else
					target->values[k] = glm::vec4(raw[k * 3], raw[k * 3 + 1], raw[k * 3 + 2], 0.0f);
			}

			if (!target->times.empty())
				cam.animation->duration = std::max(cam.animation->duration, target->times.back());
		}
	}

	cgltf_free(data);
	return scene;
}

Geometry Scene::merge() const
{
	Geometry result;
	for (const auto &entity : entities) {
		const auto &geo = geometries[entity.geometry_index];
		const uint32_t offset = static_cast <uint32_t> (result.positions.size());
		const glm::mat3 N = glm::transpose(glm::inverse(glm::mat3(entity.transform)));

		for (const auto &p : geo.positions)
			result.positions.push_back(glm::vec3(entity.transform * glm::vec4(p, 1.0f)));
		for (const auto &n : geo.normals)
			result.normals.push_back(glm::normalize(N * n));
		for (const auto &uv : geo.uvs)
			result.uvs.push_back(uv);
		for (const auto &tri : geo.triangles)
			result.triangles.push_back(tri + offset);
	}
	return result;
}

std::pair <glm::vec3, glm::vec3> Scene::bounds() const
{
	glm::vec3 bmin(FLT_MAX), bmax(-FLT_MAX);
	for (const auto &entity : entities) {
		const auto &geo = geometries[entity.geometry_index];
		for (const auto &p : geo.positions) {
			glm::vec3 wp = glm::vec3(entity.transform * glm::vec4(p, 1.0f));
			bmin = glm::min(bmin, wp);
			bmax = glm::max(bmax, wp);
		}
	}
	return { bmin, bmax };
}

glm::mat4 Camera::view_matrix() const
{
	return glm::inverse(transform);
}

glm::mat4 Camera::projection_matrix(float aspect_override) const
{
	glm::mat4 proj;
	if (type == Type::ePerspective) {
		proj = glm::perspectiveRH_ZO(fov, aspect_override, near_plane, far_plane);
	} else {
		float half_w = xmag;
		float half_h = ymag;
		proj = glm::orthoRH_ZO(-half_w, half_w, -half_h, half_h, near_plane, far_plane);
	}
	proj[1][1] *= -1.0f;
	return proj;
}

glm::vec3 Camera::eye_position() const
{
	return glm::vec3(transform[3]);
}

Camera::RayFrame Camera::ray_frame(float aspect_override) const
{
	// GLTF cameras look along -Z in their local space
	glm::vec3 origin = glm::vec3(transform[3]);
	glm::vec3 forward = -glm::normalize(glm::vec3(transform[2]));
	glm::vec3 up = glm::normalize(glm::vec3(transform[1]));
	glm::vec3 right = glm::normalize(glm::cross(forward, up));

	float vlen = std::tan(fov / 2.0f);
	return {
		.u = right * vlen * aspect_override,
		.v = up * vlen,
		.w = forward,
		.origin = origin,
	};
}
